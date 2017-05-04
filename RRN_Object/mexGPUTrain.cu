/*
 * Example of how to use the mxGPUArray API in a MEX file.  This example shows
 * how to write a MEX function that takes a gpuArray input and returns a
 * gpuArray output, e.g. B=mexFunction(A).
 *
 * Copyright 2012 The MathWorks, Inc.
 */

#include "mex.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "gpu/mxGPUArray.h"
#include "assert.h"

/*
 * Device code
 */

#define MAX_BLOCK_SIZE 512

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
    extern __shared__ int s[];
   
    float error;
    int j;
    int ind;
    int unitId = blockIdx.y*gridDim.x + blockIdx.x;

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

    // Read indices specific to this rate unit (unitId)
    int p1 = 0, p3 = 0;
    if (unitId > 0) {
       p1 = cumPreSizes[unitId-1];
       p3 = cumPreSizeSq[unitId-1];
    }
    //int p2 = (int)(cumPreSizes[unitId]+0.5);
    //int p4 = (int)(cumPreSizeSq[unitId]+0.5);
    int numPre = cumPreSizes[unitId]-p1;


    float *k = (float *)s; //[MAX_BLOCK_SIZE];
    float *ex = (float *)&k[numPre]; //[MAX_BLOCK_SIZE];

    // Read rate unit specific data into shared memory
    if (threadIdx.x < numPre) {
      ind = preInds[p1+threadIdx.x]-1;
      ex[threadIdx.x] = Ex[ind];
    }
    __syncthreads();
    error = err[unitId];

    // Compute k for this rate unit
    if (threadIdx.x < numPre) {
        k[threadIdx.x] = 0;
        for (j = 0; j < numPre; j++) {
      	    k[threadIdx.x] = k[threadIdx.x] + Ps[p3+j*numPre+threadIdx.x] * ex[j];
        }
    }
    __syncthreads();

    // Compute c and update W and P for this rate unit
    float c = 0;
    if (threadIdx.x < numPre) {
       c = 1.0;
       for (j = 0; j < numPre; j++) {
       	   c = c + k[j] * ex[j];
       }
       c = 1.0/c;
//       W[inds[threadIdx.x]*N+unitId] = W[inds[threadIdx.x]*N+unitId] - c*error*k[threadIdx.x];
       W[p1+threadIdx.x] = W[p1+threadIdx.x] - c*error*k[threadIdx.x];
//       W[p1+threadIdx.x] = ex[threadIdx.x];
       for (j = 0; j < numPre; j++) {
       	   Ps[p3+j*numPre+threadIdx.x] = Ps[p3+j*numPre+threadIdx.x]-c*k[threadIdx.x]*k[j];
       }
    }
}

/*
void __global__ RunRLS(float const * const Ex,
                       float const * const err,
                       float const * const cumPreSizes,
                       float const * const cumPreSizeSq,
                       float const * const preInds,
                       float * const Ps,
                       float * const W,
                       int const N)
{
   __shared__ float k[MAX_BLOCK_SIZE];
   __shared__ float ex[MAX_BLOCK_SIZE];
   __shared__ int inds[MAX_BLOCK_SIZE];
   __shared__ float error;
   
    int j;
    int unitId = blockIdx.y*gridDim.x + blockIdx.x;
    // Read indices specific to this rate unit (unitId)
    int p1 = 0, p3 = 0;

    if (unitId > 0) {
       p1 = (int)(cumPreSizes[unitId-1]+0.5);
       p3 = (int)(cumPreSizeSq[unitId-1]+0.5);
    }
    //int p2 = (int)(cumPreSizes[unitId]+0.5);
    //int p4 = (int)(cumPreSizeSq[unitId]+0.5);
    int numPre = (int)(cumPreSizes[unitId]+0.5)-p1;
    //int numPreSq = (int)(cumPreSizeSq[unitId]+0.5)-p3;

    // Read rate unit specific data into shared memory
    if (threadIdx.x < numPre) {
      inds[threadIdx.x] = (int)(preInds[p1+threadIdx.x]+0.5)-1;
      ex[threadIdx.x] = Ex[inds[threadIdx.x]];
      if (threadIdx.x == 0) {
      	 error = err[unitId];      
      }
    }
    __syncthreads();
    
    // Compute k for this rate unit
    if (threadIdx.x < numPre) {
      k[threadIdx.x] = 0;
      for (j = 0; j < numPre; j++) {
      	  k[threadIdx.x] = k[threadIdx.x] + Ps[p3+j*numPre+threadIdx.x] * ex[j];
      }
    }
    __syncthreads();

    // Compute c and update W and P for this rate unit
    float c = 0;
    c = 1.0;
    for (j = 0; j < numPre; j++) {
       c = c + k[j] * ex[j];
    }
    c = 1.0/c;
    
    j = threadIdx.x;
    int row, col;
    while (j < numPre*numPre) { //for (j = 0; j < numPre; j++) {
       row = j%numPre;
       col = int(j/numPre);
       //Ps[p3+j*numPre+threadIdx.x] = Ps[p3+j*numPre+threadIdx.x]-c*k[threadIdx.x]*k[j];
       Ps[p3+j] = Ps[p3+j]-c*k[row]*k[col];
       j = j + blockDim.x;
    }

    if (threadIdx.x < numPre) {
       W[inds[threadIdx.x]*N+unitId] = W[inds[threadIdx.x]*N+unitId] - c*error*k[threadIdx.x];
    }
}


            ExList = [1:numEx*0.95]; % + (stim-1)*25;
            for i=[ExList] %loop through Post Ex
               %From Sussillo and Abbott Code
               preind = PreSyn(i).ind;
               ex =  Ex(preind);
               k = PRec(i).P*ex;
               expex = ex'*k;
               c = 1.0/(1.0 + expex);
               PRec(i).P = PRec(i).P - k*(k'*c);
               dw = error_rec(i)*k*c;
               WExEx(preind,i) = WExEx(preind,i) - dw;
            end
*/

/*
 * Host code
 */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    /* Declare all variables.*/
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
    
    /* Initialize the MathWorks GPU API. */
    mxInitGPU();

    /* Throw an error if the input is not a GPU array. */
    if (nrhs!=12) {
        mexErrMsgIdAndTxt(errId, errMsg1);
    }
    else if (!(mxIsGPUArray(prhs[0])) || !(mxIsGPUArray(prhs[1])) || !(mxIsGPUArray(prhs[2])) || !(mxIsGPUArray(prhs[3])) || !(mxIsGPUArray(prhs[4])) || !(mxIsGPUArray(prhs[5])) || !(mxIsGPUArray(prhs[6])) || !(mxIsGPUArray(prhs[7]))) {
        mexErrMsgIdAndTxt(errId, errMsg2);
    }
    else if (!(mxIsSingle(prhs[8])) || !(mxIsSingle(prhs[9])) || !(mxIsSingle(prhs[10]))  || !(mxIsSingle(prhs[11]))) {
        mexErrMsgIdAndTxt(errId, errMsg3);
    }

    Ex = mxGPUCreateFromMxArray(prhs[0]);
    err = mxGPUCreateFromMxArray(prhs[1]);
    cumPreSizes = mxGPUCreateFromMxArray(prhs[2]);
    cumPreSizeSq = mxGPUCreateFromMxArray(prhs[3]);
    preInds = mxGPUCreateFromMxArray(prhs[4]);
    Ps = const_cast<mxGPUArray *>(mxGPUCreateFromMxArray(prhs[5])); // mxGPUCopyFromMxArray(prhs[5]);
    W = const_cast<mxGPUArray *>(mxGPUCreateFromMxArray(prhs[6])); // mxGPUCopyFromMxArray(prhs[6]);
    exList = mxGPUCreateFromMxArray(prhs[7]);
    int N1 = (int) (mxGetScalar(prhs[8])+0.5);
    int N2 = (int) (mxGetScalar(prhs[9])+0.5);
    int K = (int) (mxGetScalar(prhs[10])+0.5);
    int exListLen = (int) (mxGetScalar(prhs[11])+0.5);

    /* Choose a reasonably sized number of threads for the block. */
    int threadsPerBlock = K; //K; //1024; //1024; // K;
    int blocksPerGridx = N1;
    int blocksPerGridy = N2;
    dim3 blocksPerGrid(blocksPerGridx, blocksPerGridy);
    /*
     * Verify that A really is a float array before extracting the pointer.
     */
/*    if (mxGPUGetClassID(A) != mxFLOAT_CLASS) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }
*/
    /*
     * Now that we have verified the data type, extract a pointer to the input
     * data on the device.
     */
    d_Ex = (float const *)(mxGPUGetDataReadOnly(Ex));
    d_err = (float const *)(mxGPUGetDataReadOnly(err));
    d_cumPreSizes = (int const *)(mxGPUGetDataReadOnly(cumPreSizes));
    d_cumPreSizeSq = (int const *)(mxGPUGetDataReadOnly(cumPreSizeSq));
    d_preInds = (int const *)(mxGPUGetDataReadOnly(preInds));
    d_Ps = (float *)(mxGPUGetData(Ps));
    d_W = (float *)(mxGPUGetData(W));
    d_exList = (int const *)(mxGPUGetDataReadOnly(exList));;

    /*
     * Call the kernel using the CUDA runtime API. We are using a 1-d grid here,
     * and it would be possible for the number of elements to be too large for
     * the grid. For this example we are not guarding against this possibility.
     */
    RunRLS<<<blocksPerGrid, threadsPerBlock, (2*sizeof(float))*K>>>(d_Ex, d_err, d_cumPreSizes, d_cumPreSizeSq, d_preInds, d_Ps, d_W, d_exList, exListLen, N1*N2);
    cudaDeviceSynchronize();

    /* Wrap the result up as a MATLAB gpuArray for return. */
    //plhs[0] = mxGPUCreateMxArrayOnGPU(Ps);
    //plhs[1] = mxGPUCreateMxArrayOnGPU(W);

    /*
     * The mxGPUArray pointers are host-side structures that refer to device
     * data. These must be destroyed before leaving the MEX function.
     */
    mxGPUDestroyGPUArray(Ex);
    mxGPUDestroyGPUArray(err);
    mxGPUDestroyGPUArray(cumPreSizes);
    mxGPUDestroyGPUArray(cumPreSizeSq);
    mxGPUDestroyGPUArray(preInds);
    mxGPUDestroyGPUArray(Ps);
    mxGPUDestroyGPUArray(W);
    mxGPUDestroyGPUArray(exList);
}
