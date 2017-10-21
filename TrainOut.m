function TrainOut(Net)
TestSeed=Net.getNetworkSeed*31; % make sure the testing seed is different than the traing seed
Net.newState(TestSeed);
Net.newwRecOut;
numOutTrials = 10;
trigStart = 300;
trigEnd = Net.TrigDur + trigStart;
XDurr=Net.TargLen/2000; % Target hit times assumes 2000 ms target
AllTargTimes = [163,513,750,1200,1750]*XDurr+trigEnd;
OutDur = Net.TargLen+trigEnd;
Net.generateP_CPU;
OutTarget = zeros(1,OutDur);
for targTInd = 1:numel(AllTargTimes)
    thisHitTime=AllTargTimes(targTInd);
    ThisHit = normpdf(1:OutDur,thisHitTime,50);
    ThisHit=(1/max(ThisHit)).*ThisHit;
    OutTarget = OutTarget+ThisHit;
end
OutTarget=OutTarget-mean(OutTarget);
figure; plot(OutTarget);
inFig = figure; hold on; title('Input')
outFig = figure; hold on; title('Out Train')
plot(OutTarget,'--k','linewidth',2);
recFig = figure; hold on; title('RNNUnit Out Train')
for trialInd = 1:numOutTrials
    ThisOutDur = size(OutTarget,2);
    ThisTotT = ThisOutDur+500;
    outTrnWind = trigEnd:ThisOutDur;
    InPulses = Net.generateInputPulses(...
        [2, 3],...
        [Net.TrigAmp,Net.originalTonicLvl],...
        [trigStart, trigStart],...
        [trigEnd, ThisOutDur],...
        ThisTotT);
    figure(inFig); clf; plot(InPulses');
    NoiseIn = Net.generateNoiseInput(InPulses, ...
        aNoise);
    hEx = zeros(Net.nRec, ThisTotT);
    hOut = zeros(Net.numOut, ThisTotT);
    Net.randStateRRN;
    for t = 1:ThisTotT
        In = NoiseIn(:,t);
        [~, hEx(:,t)] = Net.IterateRNN_CPU(In);
        hOut(:,t) = Net.IterateOutCPU;
        if ismember(t,outTrnWind)
            Net.trainOutFORCE(OutTarget(:,t));
        end
    end
    figure(recFig); plot(hEx(10,:));
    figure(outFig); plot(hOut'); drawnow;
end
Net.clearStateVars;
end
