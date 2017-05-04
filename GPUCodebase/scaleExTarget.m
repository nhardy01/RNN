
function [scaledExTargs] = scaleTarget(target, trainStart, activeRRNEnd, InAmp2, scaleDir, scalingTics, originalTonicLvl, HighResSampleRate, scalingFactor)

trainingExTarget = target(:, trainStart+1:activeRRNEnd+trainStart);
preTrainExTarget = target(:,1:trainStart);
postActiveExTraget = target(:,activeRRNEnd+trainStart+1:end);
highResExTarget = interp1(trainingExTarget',[1/HighResSampleRate:1/HighResSampleRate:activeRRNEnd]);
for inAmpInd = 1:length(InAmp2)
    currentStim = InAmp2(inAmpInd);
    numScales = (currentStim-originalTonicLvl)/scalingTics*scaleDir;
    newExTargSample = round([1:(1/(1-scalingFactor*numScales)):activeRRNEnd]*HighResSampleRate);
    sampledExTarget = highResExTarget(newExTargSample,:);
    newExTarget = [preTrainExTarget sampledExTarget' postActiveExTraget];
    scaledExTargs{inAmpInd} = newExTarget;
end
end