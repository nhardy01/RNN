function TrainRNN_TempInv(RNN,TrainTrials)
trigStart=500;
trigEnd = RNN.TrigDur + trigStart;
innateTotT = RNN.TargLen + trigEnd + RNN.restDur;
%% Generate the RNN target
InPulses = RNN.generateInputPulses(...
    [RNN.CueIn, RNN.SpeedIn],...
    [RNN.TrigAmp, RNN.originalTonicLvl],...
    [trigStart, trigStart],...
    [trigEnd, trigEnd+RNN.TargLen],...
    innateTotT);
NoiseIn = RNN.generateNoiseInput(InPulses,0);
InnateTarg = zeros(RNN.numEx, innateTotT);
RNN.randStateRRN;
for t = 1:innateTotT
    In = NoiseIn(:,t);
    [~, InnateTarg(:,t)] = RNN.IterateRNN_CPU(In);
end
GatedTarg = RNN.gatedExTarget(InnateTarg,...
    innateTotT-RNN.restDur, 30);
[ScaledTargs, TargLens] = RNN.scaleTarget(GatedTarg,...
    trigEnd, trigEnd+RNN.TargLen);
clear GatedTarg InnateTarg InPulses NoiseIn
%% Train RNN
RNN.setRNNTarget(ScaledTargs);
RNN.generate_W_P_GPU;
for trial = 1:TrainTrials
    stim = RNN.TrainStimOrder(trial);
    thisTrialTrainTime = TargLens(stim);
    sigEnd = TargLens(stim) - RNN.restDur;
    thisTarg = gpuArray(single(ScaledTargs{stim}));
    thisSpeedSig = RNN.ExExTrainTonicStims(stim);
    InPulses = RNN.generateInputPulses(...
        [RNN.CueIn,RNN.SpeedIn],...
        [RNN.TrigAmp, thisSpeedSig],...
        [trigStart, trigStart],...
        [trigEnd, sigEnd],...
        thisTrialTrainTime);
    NoiseIn = RNN.generateNoiseInput(InPulses,RNN.innateNoiseLvl);
    RNN.randStateRRN;
    RNN.RNNStateGPU;
    RNN.trainRNNTargetGPU(thisTarg,...
        [trigEnd:RNN.trainTimeStep:thisTrialTrainTime], NoiseIn);
    fprintf([RNN.getName,' TrainTrial:%i/%i Stim:%i/%i SpeedIn:%.2g SpeedDur:%i\n'],...
        trial,TrainTrials,stim,numel(unique(RNN.TrainStimOrder)),thisSpeedSig,sigEnd-trigEnd)
end

RNN.reconWs; % reconstruct weights from GPU values
RNN.clearStateVars;
end