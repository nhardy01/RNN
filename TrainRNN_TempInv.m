function TrainRNN_TempInv(Net,durBL,TrainTrials)
trigStart = 500;
TrigDur = Net.tau * 5;
trigEnd = TrigDur + trigStart;
ixCue = 2;
ixSpeed = 3;
aInBL = 0.3;
durFact = [1,4];
aIn = aInBL./durFact;
aCue = 3;
tRest = 0;
stimOrder = 1:TrainTrials;
stimOrder(2:2:end) = 2;
aNoise = 0.25;
tTrainStep = 5;
innateTotT = durBL + trigEnd + tRest;
%% Generate the Net target
InPulses = Net.generateInputPulses(...
    [ixCue, ixSpeed],...
    [aCue, aInBL],...
    [trigStart, trigStart],...
    [trigEnd, trigEnd+durBL],...
    innateTotT);
NoiseIn = Net.generateNoiseInput(InPulses,0);
InnateTarg = zeros(Net.nRec, innateTotT);
Net.randStateRRN;
for t = 1:innateTotT
    In = NoiseIn(:,t);
    [~, InnateTarg(:,t)] = Net.IterateRNN_CPU(In);
end
GatedTarg = Net.gatedRecTarget(InnateTarg,...
    innateTotT-tRest, 30);
[ScaledTargs, TargLens] = Net.scaleTarget(InnateTarg,...
    trigEnd, trigEnd+durBL,1,100,aIn,aInBL);
clear GatedTarg InnateTarg InPulses NoiseIn
%% Train Net
Net.setRNNTarget(ScaledTargs);
Net.generate_W_P_GPU;
for trial = 1:TrainTrials
    stim = stimOrder(trial);
    thisTrialTrainTime = TargLens(stim);
    sigEnd = TargLens(stim) - tRest;
    thisTarg = gpuArray(single(ScaledTargs{stim}));
    thisSpeedSig = aIn(stim);
    InPulses = Net.generateInputPulses(...
        [ixCue,ixSpeed],...
        [aCue, thisSpeedSig],...
        [trigStart, trigStart],...
        [trigEnd, sigEnd],...
        thisTrialTrainTime);
    NoiseIn = Net.generateNoiseInput(InPulses,aNoise);
    Net.randStateRRN;
    Net.RNNStateGPU;
    Net.trainRNNFORCE_GPU(thisTarg,...
        [trigEnd:tTrainStep:thisTrialTrainTime], NoiseIn);
    fprintf([Net.getName,' TrainTrial:%i/%i Stim:%i/%i SpeedIn:%.2g SpeedDur:%i\n'],...
        trial,TrainTrials,stim,numel(unique(stimOrder)),thisSpeedSig,sigEnd-trigEnd)
end

Net.reconWs; % reconstruct weights from GPU values
Net.clearStateVars;
end
