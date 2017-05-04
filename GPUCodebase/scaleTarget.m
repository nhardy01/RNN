
function [scaledExTargs, times] = scaleTarget(target, trainStart, activeRRNEnd, InAmp2, scaleDir, scalingTics, originalTonicLvl, HighResSampleRate, scalingFactor)

numScaledTargs = length(InAmp2);
times = zeros(1,numScaledTargs);
trainingExTarget = target(:, trainStart+1:activeRRNEnd);
preTrainExTarget = target(:,1:trainStart);
postActiveExTraget = target(:,activeRRNEnd+1:end);
highResExTarget = interp1(trainingExTarget',[1/HighResSampleRate:1/HighResSampleRate:activeRRNEnd-trainStart]);
if size(highResExTarget,1) > 1
    highResExTarget = highResExTarget';
end
for inAmpInd = 1:numScaledTargs
    currentStim = InAmp2(inAmpInd);
    numScales = (currentStim-originalTonicLvl)/scalingTics*scaleDir;
    newExTargSample = round([1:(1/(1-scalingFactor*numScales)):activeRRNEnd-trainStart]*HighResSampleRate);
    sampledExTarget = highResExTarget(:, newExTargSample);
    newExTarget = [preTrainExTarget sampledExTarget postActiveExTraget];
    scaledExTargs{inAmpInd} = newExTarget;
    times(inAmpInd) = size(newExTarget,2);
end
end