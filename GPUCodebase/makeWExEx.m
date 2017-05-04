%{
Initiates sparse, random RRN connectivity matrix with no autapses (numEx x
numEx)
Accepts P_Connect sparsity variable, g = gain vaiable
Requires global numEx
returns WExEx
%}

function WExEx = makeWExEx(P_Connect, g, numEx)
WMask = rand(numEx,numEx);
WMask(WMask<(1-P_Connect))=0;
WMask(WMask>0) = 1;
WExEx = randn(numEx,numEx)*sqrt(1/(numEx*P_Connect));
WExEx = g*WExEx.*WMask;
WExEx(1:(numEx+1):numEx*numEx)=0;
end