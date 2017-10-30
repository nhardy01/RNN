#include "mex.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "gpu/mxGPUArray.h"
#include "math.h"

/*
 * Device code
 */
// This kernel updates the output variable. It is a bit inefficient, 
// but shouldn't change things significantly
void __global__ UpdateOut(float const * const WExOut, 
     		          float * const Out, 
			  float * const Ex, 
			  int const numOut, 
			  int const numEx)
{
   int ind = blockDim.x * blockIdx.x + threadIdx.x;
   int j;
   float tmp;

   if (ind < numOut) {
      tmp = 0;
      for (j = 0; j < numEx; j++) {
      	  tmp = tmp + WExOut[j+ind*numEx]*Ex[j];
      }
      Out[ind] = tmp;
   }
}

//This kernel applies the non-linear transform to ExV and updates Ex
//It had to be split from the rest of the update to avoid race conditions
void __global__ finishUpdateEx(float * const Ex,
                       float * const ExV,
		       int numEx)
{
    int unitId = blockIdx.x*blockDim.x + threadIdx.x;
    if (unitId < numEx) {
        Ex[unitId] = tanh(ExV[unitId]);
    }
}

// This kernel updates ExV
// Primarily it has been optimized for cache locality (minimum cache misses)
void __global__ UpdateEx(float const * const WExEx,
                       float const * const noisePlusIn,
                       int const * const cumPreSizes,
                       int const * const preInds,
                       float * const Ex,
                       float * const ExV,
                       int const numEx,
                       int const t,
                       float const tau,
		       int const upperLim,
		       int const lowerLim)
{
    extern __shared__ int s[]; // Amount of shared memory allocated is specified when the kernel is called (see below)
    int j;
    int ind;

    int unitId = blockIdx.y*gridDim.x + blockIdx.x;
    int p1 = 0;
    if (unitId > 0) {
       p1 = cumPreSizes[unitId-1];
    }
    int numPre = cumPreSizes[unitId]-p1;

    // Assign shared memory
    float *ex = (float *)s;
    float *tmpVals = (float *)&ex[numPre];

    // Read pre-synatic activity into shared memory (ex)
    if (threadIdx.x < numPre) {
      ind = preInds[p1+threadIdx.x]-1;
      ex[threadIdx.x] = Ex[ind];
    }
    __syncthreads();

    int lim = upperLim>>1;
    if (threadIdx.x < numPre) {
       tmpVals[threadIdx.x] = WExEx[p1+threadIdx.x]*ex[threadIdx.x];
       __syncthreads();
       // Efficient manual summation of each pre-synaptic unit's contribution to total current
       // This is efficient because it is parallelized over the threads in the block       
       if (numPre < upperLim) {
       	  if (threadIdx.x >= lowerLim) {
       	     tmpVals[threadIdx.x-lowerLim] = tmpVals[threadIdx.x-lowerLim] + tmpVals[threadIdx.x];
	  }
	  lim = lowerLim>>1;
       }
       if (numPre > upperLim) {
       	  if (threadIdx.x >= upperLim) {
       	     tmpVals[threadIdx.x-upperLim] = tmpVals[threadIdx.x-upperLim] + tmpVals[threadIdx.x];
	  }
       }
       for (j = lim; j >= 1; j >>= 1) {
       	   __syncthreads();       
       	   if (threadIdx.x < j) {
              tmpVals[threadIdx.x] += tmpVals[threadIdx.x+j];
    	   }
       }
    }    
    __syncthreads();       

    float tmp, tmp2;
    // Thread 0 finishes up the calculation for update of ExV
    if (threadIdx.x == 0) {
       tmp = tmpVals[0] + noisePlusIn[unitId+(t-1)*numEx];    
       tmp2 = ExV[unitId];
       tmp2 = tmp2 + (-1.0*tmp2 + tmp)/tau;
       ExV[unitId] = tmp2;
    }
}

/*
 * Host code
 */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{

    // Declare all variables
    mxGPUArray const *WExEx;
    mxGPUArray const *WExOut;
    mxGPUArray const *noisePlusIn;
    mxGPUArray const *cumPreSizes;
    mxGPUArray const *preInds;
    mxGPUArray *Ex;
    mxGPUArray *ExV;
    mxGPUArray *Out;

    float const *d_WExEx;
    float const *d_WExOut;
    float const *d_noisePlusIn;
    int const *d_cumPreSizes;
    int const *d_preInds;
    float *d_Ex;
    float *d_ExV;
    float *d_Out;

    char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const * const errMsg1 = "Incorrect number of inputs to MEX file.";
    char const * const errMsg2 = "Invalid matrix input to MEX file.";
    char const * const errMsg3 = "Invalid scalar input to MEX file.";
    
    // Initialize the MathWorks GPU API.
    mxInitGPU();

    /* Throw an error if the number of inputs is incorrect. */
    if (nrhs!=14) {
        mexErrMsgIdAndTxt(errId, errMsg1);
    }
    /* Throw an error if the vector inputs are not GPU arrays. */
    else if (!(mxIsGPUArray(prhs[0])) || !(mxIsGPUArray(prhs[1])) || !(mxIsGPUArray(prhs[2])) || !(mxIsGPUArray(prhs[3])) || !(mxIsGPUArray(prhs[4])) || !(mxIsGPUArray(prhs[5])) || !(mxIsGPUArray(prhs[6])) || !(mxIsGPUArray(prhs[7]))) {
        mexErrMsgIdAndTxt(errId, errMsg2);
    }
    /* Throw an error if the input variables are not single precision variables. */
    else if (!(mxIsSingle(prhs[8])) || !(mxIsSingle(prhs[9])) || !(mxIsSingle(prhs[10]))  || !(mxIsSingle(prhs[11]))  || !(mxIsSingle(prhs[12]))  || !(mxIsSingle(prhs[13]))) {
        mexErrMsgIdAndTxt(errId, errMsg3);
    }

    WExEx = mxGPUCreateFromMxArray(prhs[0]); // First input is recurrent weights
    WExOut = mxGPUCreateFromMxArray(prhs[1]); // Second input is output weights
    noisePlusIn = mxGPUCreateFromMxArray(prhs[2]); // Third input is noise+Input value for each unit
    cumPreSizes = mxGPUCreateFromMxArray(prhs[3]); // Fourth - presynaptic metadata
    preInds = mxGPUCreateFromMxArray(prhs[4]); // Fifth - presynaptic indices
    Ex = const_cast<mxGPUArray *>(mxGPUCreateFromMxArray(prhs[5])); // Sixth - Rate var for each unit
    ExV = const_cast<mxGPUArray *>(mxGPUCreateFromMxArray(prhs[6])); // Seventh - Voltage var for each unit
    Out = const_cast<mxGPUArray *>(mxGPUCreateFromMxArray(prhs[7])); // Eight - Output vector

    int t = (int) (mxGetScalar(prhs[8])+0.5); // Night - time
    int numEx = (int) (mxGetScalar(prhs[9])+0.5); // Tenth - number of rate units (this is corrected later)
    int cellsPerGridCol = (int) (mxGetScalar(prhs[10])+0.5); // Eleventh - number of rate units per grid column (see below)
    int numOut = (int) (mxGetScalar(prhs[11])+0.5); // Twelfth - number of outputs
    float tau = mxGetScalar(prhs[12]); // Thirteenth - time constant

    // Firing rate units organized in a 2D grid, this may not be necessary. In any case, the grid has 'cellsPerGridCol' columns, and numEx/'cellsPerGridCol'.
    // Each SM (symmetric multiprocessor) will typically simultaneously execute all threads for a given grid block or rate unit (see below), and simultaneously do this for a few rate units at a time.
    int N1 = numEx; // Dimension 1 of grid arrangement of rate units
    int N2 = cellsPerGridCol;  // Dimension 2 of grid arrangement of rate units
 
    // One thread is being assigned per synapse here. The threads within a grid block, i.e. for a given rate unit, will need to talk to each other.
    // Since number of threads is fixed for all units, this should be set to the maximum # pr pre-synaptic inputs across all rate units.
    int K = (int) (mxGetScalar(prhs[13])+0.5); // Fourteenth - number of threads per grid block
    int upperLim = (int)log2((float)K);
    int lowerLim = (int)pow(2,upperLim-1);
    upperLim = (int)pow(2,upperLim);
    numEx = numEx*cellsPerGridCol;
    
    // Note that there are hard, GPU architecture-based limits on number of threads per grid block (1024) and the dimensions of the block grid. 
    // Besides this there is also a limit on the GPU memory!
    int threadsPerBlock = K;
    int blocksPerGridx = N1;
    int blocksPerGridy = N2;
    dim3 blocksPerGrid2D(blocksPerGridx, blocksPerGridy);
    
    // Extract pointers to the input data on the device.
    d_WExEx = (float const *)(mxGPUGetDataReadOnly(WExEx));
    d_WExOut = (float const *)(mxGPUGetDataReadOnly(WExOut));
    d_noisePlusIn = (float const *)(mxGPUGetDataReadOnly(noisePlusIn));
    d_cumPreSizes = (int const *)(mxGPUGetDataReadOnly(cumPreSizes));
    d_preInds = (int const *)(mxGPUGetDataReadOnly(preInds));
    d_Ex = (float *)(mxGPUGetData(Ex));
    d_ExV = (float *)(mxGPUGetData(ExV));
    d_Out = (float *)(mxGPUGetData(Out));

    // Call CUDA kernel to run partial update on the unit rates
    UpdateEx<<<blocksPerGrid2D, threadsPerBlock, 2*sizeof(float)*K>>>(d_WExEx, 
                                                   d_noisePlusIn, 
						   d_cumPreSizes,
						   d_preInds,
						   d_Ex, 
						   d_ExV, 
						   numEx, 
						   t, 
						   tau,
						   upperLim,
						   lowerLim);
    cudaDeviceSynchronize();

    // Reorganize kernel computation dimensions a bit and call CUDA kernel to complete update on the unit rates
    threadsPerBlock = N1;
    int blocksPerGrid = N2;
    finishUpdateEx<<<blocksPerGrid, threadsPerBlock>>>(d_Ex, d_ExV, numEx);
    cudaDeviceSynchronize();
    
    // Reorganize kernel computation dimensions again and call CUDA kernel to update output unit values
    threadsPerBlock = numOut;
    blocksPerGrid = 1;
    UpdateOut<<<blocksPerGrid, threadsPerBlock>>>(d_WExOut, d_Out, d_Ex, numOut, numEx);
    cudaDeviceSynchronize();

    // Cleanup
    mxGPUDestroyGPUArray(WExEx);
    mxGPUDestroyGPUArray(WExOut);
    mxGPUDestroyGPUArray(noisePlusIn);
    mxGPUDestroyGPUArray(cumPreSizes);
    mxGPUDestroyGPUArray(preInds);
    mxGPUDestroyGPUArray(Ex);
    mxGPUDestroyGPUArray(ExV);
    mxGPUDestroyGPUArray(Out);
}
