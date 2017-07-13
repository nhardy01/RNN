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
#include "math.h"

/*
 * Device code
 */
#define MAX_BLOCK_SIZE  512 //4048

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
/*
      %%% UPDATE Ex & Out UNITS
      Out = gWExOut'*Ex;
*/

void __global__ finishUpdateEx(float * const Ex,
                       float * const ExV,
		       int numEx)
{
    int unitId = blockIdx.x*blockDim.x + threadIdx.x;
    if (unitId < numEx) {
    Ex[unitId] = tanh(ExV[unitId]);
    }
}


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
    extern __shared__ int s[];
    //__shared__ float ex[MAX_BLOCK_SIZE];
    //__shared__ float tmpVals[MAX_BLOCK_SIZE];    
    int j;
    int ind;

    int unitId = blockIdx.y*gridDim.x + blockIdx.x;
    int p1 = 0;
    if (unitId > 0) {
       p1 = cumPreSizes[unitId-1];
    }
    int numPre = cumPreSizes[unitId]-p1;
    float *ex = (float *)s; //[MAX_BLOCK_SIZE];
    float *tmpVals = (float *)&ex[numPre]; //[MAX_BLOCK_SIZE];

     // Read rate unit specific data into shared memory
    if (threadIdx.x < numPre) {
      ind = preInds[p1+threadIdx.x]-1;
      ex[threadIdx.x] = Ex[ind];
    }
    __syncthreads();

    int lim = upperLim>>1;
    if (threadIdx.x < numPre) {
       tmpVals[threadIdx.x] = WExEx[p1+threadIdx.x]*ex[threadIdx.x];
       __syncthreads();       
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
    if (threadIdx.x == 0) {
       tmp = tmpVals[0] + noisePlusIn[unitId+(t-1)*numEx];    
       tmp2 = ExV[unitId];
       tmp2 = tmp2 + (-1.0*tmp2 + tmp)/tau;
       ExV[unitId] = tmp2;
    }
}

/*
            ex_input = gWExEx*Ex + gWInEx*In(:,t) + gnoiseArr(:,t);
      ExV = ExV + (-ExV + ex_input)./tau;
      Ex = tanh(ExV);
*/

/*
 * Host code
 */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{

    /* Declare all variables.*/
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
    
    /* Initialize the MathWorks GPU API. */
    mxInitGPU();

    /* Throw an error if the input is not a GPU array. */
    if (nrhs!=14) {
        mexErrMsgIdAndTxt(errId, errMsg1);
    }
    else if (!(mxIsGPUArray(prhs[0])) || !(mxIsGPUArray(prhs[1])) || !(mxIsGPUArray(prhs[2])) || !(mxIsGPUArray(prhs[3])) || !(mxIsGPUArray(prhs[4])) || !(mxIsGPUArray(prhs[5])) || !(mxIsGPUArray(prhs[6])) || !(mxIsGPUArray(prhs[7]))) {
        mexErrMsgIdAndTxt(errId, errMsg2);
    }
    else if (!(mxIsSingle(prhs[8])) || !(mxIsSingle(prhs[9])) || !(mxIsSingle(prhs[10]))  || !(mxIsSingle(prhs[11]))  || !(mxIsSingle(prhs[12]))  || !(mxIsSingle(prhs[13]))) {
        mexErrMsgIdAndTxt(errId, errMsg3);
    }

    WExEx = mxGPUCreateFromMxArray(prhs[0]);
    WExOut = mxGPUCreateFromMxArray(prhs[1]);
    noisePlusIn = mxGPUCreateFromMxArray(prhs[2]);
    cumPreSizes = mxGPUCreateFromMxArray(prhs[3]);
    preInds = mxGPUCreateFromMxArray(prhs[4]);    
    Ex = const_cast<mxGPUArray *>(mxGPUCreateFromMxArray(prhs[5]));
    ExV = const_cast<mxGPUArray *>(mxGPUCreateFromMxArray(prhs[6]));
    Out = const_cast<mxGPUArray *>(mxGPUCreateFromMxArray(prhs[7]));

    int t = (int) (mxGetScalar(prhs[8])+0.5);
    int numEx = (int) (mxGetScalar(prhs[9])+0.5);
    int cellsPerGridCol = (int) (mxGetScalar(prhs[10])+0.5);
    int numOut = (int) (mxGetScalar(prhs[11])+0.5);
    float tau = mxGetScalar(prhs[12]);

    int N1 = numEx;
    int N2 = cellsPerGridCol;
    int K = (int) (mxGetScalar(prhs[13])+0.5);
    int upperLim = (int)log2((float)K);
    int lowerLim = (int)pow(2,upperLim-1);
    upperLim = (int)pow(2,upperLim);
    numEx = numEx*cellsPerGridCol;
    
    /* Choose a reasonably sized number of threads for the block. */
    int threadsPerBlock = K; //K; //1024; //1024; // K;
    int blocksPerGridx = N1;
    int blocksPerGridy = N2;
    dim3 blocksPerGrid2D(blocksPerGridx, blocksPerGridy);
    
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
    d_WExEx = (float const *)(mxGPUGetDataReadOnly(WExEx));
    d_WExOut = (float const *)(mxGPUGetDataReadOnly(WExOut));
    d_noisePlusIn = (float const *)(mxGPUGetDataReadOnly(noisePlusIn));
    d_cumPreSizes = (int const *)(mxGPUGetDataReadOnly(cumPreSizes));
    d_preInds = (int const *)(mxGPUGetDataReadOnly(preInds));
    d_Ex = (float *)(mxGPUGetData(Ex));
    d_ExV = (float *)(mxGPUGetData(ExV));
    d_Out = (float *)(mxGPUGetData(Out));

    /*
     * Call the kernel using the CUDA runtime API. We are using a 1-d grid here,
     * and it would be possible for the number of elements to be too large for
     * the grid. For this example we are not guarding against this possibility.
     */
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
    //cudaDeviceSynchronize();

    threadsPerBlock = N1;
    int blocksPerGrid = N2;
    finishUpdateEx<<<blocksPerGrid, threadsPerBlock>>>(d_Ex, d_ExV, numEx);
    //cudaDeviceSynchronize();
    
    /* Wrap the result up as a MATLAB gpuArray for return. */
    //plhs[0] = mxGPUCreateMxArrayOnGPU(Ps);
    //plhs[1] = mxGPUCreateMxArrayOnGPU(W);

    threadsPerBlock = numOut;
    blocksPerGrid = 1;
    UpdateOut<<<blocksPerGrid, threadsPerBlock>>>(d_WExOut, d_Out, d_Ex, numOut, numEx);
    cudaDeviceSynchronize();

    /*
     * The mxGPUArray pointers are host-side structures that refer to device
     * data. These must be destroyed before leaving the MEX function.
     */
    mxGPUDestroyGPUArray(WExEx);
    mxGPUDestroyGPUArray(WExOut);
    mxGPUDestroyGPUArray(noisePlusIn);
    mxGPUDestroyGPUArray(cumPreSizes);
    mxGPUDestroyGPUArray(preInds);
    mxGPUDestroyGPUArray(Ex);
    mxGPUDestroyGPUArray(ExV);
    mxGPUDestroyGPUArray(Out);
}
