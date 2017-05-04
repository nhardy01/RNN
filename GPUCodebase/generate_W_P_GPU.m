

function [PRecs, PIndsC, cumPreSizesC, cumPreSizeSq, mxPre, gWExEx, gWExOut, cumPreSizes, PInds, P] = generate_W_P_GPU(WExEx, numEx, alphaParam, WExOut)
PRecs = [];
PIndsC = [];
cumPreSizesC = [];
cumPreSizeSq = 0;
mxPre = 0;
Ws = [];
for i=1:numEx
    ind = find(WExEx(:,i))';
    PIndsC = [PIndsC ind];
    Ws = [Ws WExEx(ind,i)'];
    cumPreSizesC = [cumPreSizesC length(PIndsC)];
    if length(ind) > mxPre
        mxPre = length(ind);
    end
    cumPreSizeSq = cumPreSizeSq + length(ind)^2;
end
PRecs = zeros(1, cumPreSizeSq);
cumPreSizeSq=[];
stP = 1;

for i=1:numEx
    if i > 1
        en = cumPreSizesC(i) - cumPreSizesC(i-1);
    else
        en = cumPreSizesC(i);
    end
    PTmp = eye(en)/alphaParam;
    PRecs(stP:(stP+en^2-1)) = PTmp(:)';
    cumPreSizeSq = [cumPreSizeSq stP+en^2-1];
    stP = stP+en^2;
    [i stP-1 length(PRecs)];
end
PRecs = gpuArray(single(PRecs));
PInds = gpuArray(int32(PIndsC));
cumPreSizes = gpuArray(int32(cumPreSizesC));
cumPreSizeSq = gpuArray(int32(cumPreSizeSq));
P = gpuArray(single(eye(numEx)/alphaParam));
gWExEx = gpuArray(single(Ws));
gWExOut = gpuArray(single(WExOut));
end