#include "mex.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "gpu/mxGPUArray.h"
#include "assert.h"

/*
 * Device code
 */
// This kernel update the recurrent weights and P matrices for all rate units
// Primarily it has been optimized for cache locality (minimum cache misses)
void __global__ RunRLS(float const * const Ex,
                       float const * const err,
                       int const * const cumPreSizes,
                       int const * const cumPreSizeSq,
                       int const * const preInds,
                       float * const Ps,
                       float * const W,
		       int const * const exList,
		       int const exListLen,
                       int const N)
{
    extern __shared__ int s[];  // Amount of shared memory allocated is specified when the kernel is called (see below)
   
    float error;
    int j;
    int ind;
    int unitId = blockIdx.y*gridDim.x + blockIdx.x;

    // Parallel search to check if this unit is on the list of units to be trained
    j = threadIdx.x;
    if (j == 0) {
       s[0] = 0;
    }
    __syncthreads();
    while (j < exListLen) {
    	  if (exList[j] == (unitId+1)) {
	     s[0] = 1;
	  }
	  j = j + blockDim.x;
    }
    __syncthreads();
    if (s[0] == 0) {
       return;
    }

    // Read lookup metadata specific to this rate unit (unitId)
    int p1 = 0, p3 = 0;
    if (unitId > 0) {
       p1 = cumPreSizes[unitId-1];
       p3 = cumPreSizeSq[unitId-1];
    }
    int numPre = cumPreSizes[unitId]-p1;

    // Assign shared memory
    float *k = (float *)s;
    float *ex = (float *)&k[numPre];

    // Read pre-synatic activity into shared memory (ex)
    if (threadIdx.x < numPre) {
      ind = preInds[p1+threadIdx.x]-1;
      ex[threadIdx.x] = Ex[ind];
    }
    __syncthreads();
    error = err[unitId];

    // Compute veci=tor k for this rate unit in parallel
    if (threadIdx.x < numPre) {
        k[threadIdx.x] = 0;
        for (j = 0; j < numPre; j++) {
      	    k[threadIdx.x] = k[threadIdx.x] + Ps[p3+j*numPre+threadIdx.x] * ex[j];
        }
    }
    __syncthreads();

    // Compute c and update W and P for this rate unit in parallel
    float c = 0;
    if (threadIdx.x < numPre) {
       c = 1.0;
       for (j = 0; j < numPre; j++) {
       	   c = c + k[j] * ex[j];
       }
       c = 1.0/c;
       W[p1+threadIdx.x] = W[p1+threadIdx.x] - c*error*k[threadIdx.x];
       for (j = 0; j < numPre; j++) {
       	   Ps[p3+j*numPre+threadIdx.x] = Ps[p3+j*numPre+threadIdx.x]-c*k[threadIdx.x]*k[j];
       }
    }
}

/*
 * Host code
 */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    // Declare all variables
    mxGPUArray const *Ex;
    mxGPUArray const *err;
    mxGPUArray const *cumPreSizes;
    mxGPUArray const *cumPreSizeSq;
    mxGPUArray const *preInds;
    mxGPUArray const *exList;
    mxGPUArray *Ps;
    mxGPUArray *W;

    float const *d_Ex;
    float const *d_err;
    int const *d_cumPreSizes;
    int const *d_cumPreSizeSq;
    int const *d_preInds;
    int const *d_exList;
    float *d_Ps;
    float *d_W;

    char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const * const errMsg1 = "Incorrect number of inputs to MEX file.";
    char const * const errMsg2 = "Invalid matrix input to MEX file.";
    char const * const errMsg3 = "Invalid scalar input to MEX file.";
    
    // Initialize the MathWorks GPU API.
    mxInitGPU();

    /* Throw an error if the number of inputs is incorrect. */
    if (nrhs!=12) {
        mexErrMsgIdAndTxt(errId, errMsg1);
    }
    /* Throw an error if the vector inputs are not GPU arrays. */
    else if (!(mxIsGPUArray(prhs[0])) || !(mxIsGPUArray(prhs[1])) || !(mxIsGPUArray(prhs[2])) || !(mxIsGPUArray(prhs[3])) || !(mxIsGPUArray(prhs[4])) || !(mxIsGPUArray(prhs[5])) || !(mxIsGPUArray(prhs[6])) || !(mxIsGPUArray(prhs[7]))) {
        mexErrMsgIdAndTxt(errId, errMsg2);
    }
    /* Throw an error if the input variables are not single precision variables. */
    else if (!(mxIsSingle(prhs[8])) || !(mxIsSingle(prhs[9])) || !(mxIsSingle(prhs[10]))  || !(mxIsSingle(prhs[11]))) {
        mexErrMsgIdAndTxt(errId, errMsg3);
    }

    Ex = mxGPUCreateFromMxArray(prhs[0]); // First input is rate of recurrent units
    err = mxGPUCreateFromMxArray(prhs[1]); // Second input is the error in the recurrent unit rates
    cumPreSizes = mxGPUCreateFromMxArray(prhs[2]); // Third - presynaptic metadata
    cumPreSizeSq = mxGPUCreateFromMxArray(prhs[3]); // Fourth - P-matrix metadata
    preInds = mxGPUCreateFromMxArray(prhs[4]); // Fifth - presynaptic indices
    Ps = const_cast<mxGPUArray *>(mxGPUCreateFromMxArray(prhs[5])); // Sixth - P matrices in concatenated vector form
    W = const_cast<mxGPUArray *>(mxGPUCreateFromMxArray(prhs[6])); // Seventh - WExEx matrix in vector form
    exList = mxGPUCreateFromMxArray(prhs[7]); // Eight - List of rate units to be trained
    // Firing rate units organized in a 2D grid, this may not be necessary. In any case, the grid has 'cellsPerGridCol' columns, and numEx/'cellsPerGridCol'.
    // Each SM (symmetric multiprocessor) will typically simultaneously execute all threads for a given grid block or rate unit (see below), and simultaneously do this for a few rate units at a time.
    int N1 = (int) (mxGetScalar(prhs[8])+0.5); // Ninth - Dimension 1 of grid arrangement of rate units
    int N2 = (int) (mxGetScalar(prhs[9])+0.5); // Tenth - Dimension 2 of grid arrangement of rate units
    // One thread is being assigned per synapse here. The threads within a grid block, i.e. for a given rate unit, will need to talk to each other.
    // Since number of threads is fixed for all units, this should be set to the maximum # pr pre-synaptic inputs across all rate units.
    int K = (int) (mxGetScalar(prhs[10])+0.5); // Eleventh - number of threads per grid block
    int exListLen = (int) (mxGetScalar(prhs[11])+0.5); // Twelfth - size of list of rate units to be trained

    // Note that there are hard, GPU architecture-based limits on number of threads per grid block (1024) and the dimensions of the block grid. 
    // Besides this there is also a limit on the GPU memory!
    int threadsPerBlock = K;
    int blocksPerGridx = N1;
    int blocksPerGridy = N2;
    dim3 blocksPerGrid(blocksPerGridx, blocksPerGridy);

    // Extract pointers to the input data on the device.
    d_Ex = (float const *)(mxGPUGetDataReadOnly(Ex));
    d_err = (float const *)(mxGPUGetDataReadOnly(err));
    d_cumPreSizes = (int const *)(mxGPUGetDataReadOnly(cumPreSizes));
    d_cumPreSizeSq = (int const *)(mxGPUGetDataReadOnly(cumPreSizeSq));
    d_preInds = (int const *)(mxGPUGetDataReadOnly(preInds));
    d_Ps = (float *)(mxGPUGetData(Ps));
    d_W = (float *)(mxGPUGetData(W));
    d_exList = (int const *)(mxGPUGetDataReadOnly(exList));;

    // Call CUDA kernel to run RLS update on all synapses
    RunRLS<<<blocksPerGrid, threadsPerBlock, (2*sizeof(float))*K>>>(d_Ex, d_err, d_cumPreSizes, d_cumPreSizeSq, d_preInds, d_Ps, d_W, d_exList, exListLen, N1*N2);
    cudaDeviceSynchronize();


    // Cleanup
    mxGPUDestroyGPUArray(Ex);
    mxGPUDestroyGPUArray(err);
    mxGPUDestroyGPUArray(cumPreSizes);
    mxGPUDestroyGPUArray(cumPreSizeSq);
    mxGPUDestroyGPUArray(preInds);
    mxGPUDestroyGPUArray(Ps);
    mxGPUDestroyGPUArray(W);
    mxGPUDestroyGPUArray(exList);
}
