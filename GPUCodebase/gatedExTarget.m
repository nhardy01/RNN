

function [gatedTarget] = gatedExTarget(EXTARGET, restTime, tau, numEx, ZeroTime)

totalTargLen = restTime+ZeroTime;
ExTargetMask = ones(1,totalTargLen);
ExTargetMask(ZeroTime:restTime+ZeroTime)= exp(-([ZeroTime:totalTargLen]-ZeroTime)/(2*tau));
ExTargetMask(ExTargetMask>1)=1;
gatedTarget = EXTARGET(:,1:totalTargLen).*repmat(ExTargetMask,numEx,1);

end