close all
clear
InFig = figure; NoiseFig = figure; TargFig = figure; TrainFig = figure;
%% Standard Parameters
Seed = 110;
TargLen = 4000;
TrigAmp = 5;
TrigStart = 500;
restDur = 30000;
N = 700;
TrainTrials = 10;
Tau = 50;
G = 1.6;
TrigDur = Tau*5;
NoiseAmp=0.05;
ScaleF=1;
StimOrder=ones(1,TrainTrials);
%% Run
MasterSaveDir = '~/Documents/Data/TemporalInvariance/GatingOnly/';
%% Set up save folder
SaveDir = fullfile(MasterSaveDir);
if ~exist(SaveDir)
    mkdir(SaveDir)
end
ThisNet = ThisNet(Seed,G,N,2,1); %initiate network
ThisNet.TargLen = TargLen;
ThisNet.tau = Tau;
ThisNet.TrigDur = TrigDur;
ThisNet.TrigAmp = TrigAmp;
ThisNet.TrainStimOrder = StimOrder;
ThisNet.TrigAmp = TrigAmp;
ThisNet.TrigDur = TrigDur;
ThisNet.innateNoiseLvl = NoiseAmp;
ThisNet.scalingFactor = ScaleF;
ThisNet.originalTonicLvl=0;
ThisNet.ExExTrainTonicStims=0;
ThisNet.restDur=restDur;
ThisNet.trainTimeStep=5;
ThisNet.SpeedIn=NaN;

trigEnd = ThisNet.TrigDur + TrigStart;
innateTotT = ThisNet.TargLen + trigEnd + ThisNet.restDur;
%% Generate the ThisNet target
InPulses = ThisNet.generateInputPulses(...
    [ThisNet.CueIn],...
    [ThisNet.TrigAmp],...
    [TrigStart],...
    [trigEnd],...
    innateTotT);
NoiseIn = ThisNet.generateNoiseInput(InPulses,0);
InnateTarg = zeros(ThisNet.numEx, innateTotT);
ThisNet.randStateRRN;
for t = 1:innateTotT
    In = NoiseIn(:,t);
    [~, InnateTarg(:,t)] = ThisNet.IterateRNN_CPU(In);
end
GatedTarg = ThisNet.gatedExTarget(InnateTarg,...
    innateTotT-ThisNet.restDur, 30);
clear InnateTarg InPulses NoiseIn
%% Train ThisNet
ThisNet.setRNNTarget(GatedTarg);
ThisNet.generate_W_P_GPU;
thisTarg = gpuArray(GatedTarg);
for trial = 1:TrainTrials
    stim = ThisNet.TrainStimOrder(trial);
    InPulses = ThisNet.generateInputPulses(...
        [ThisNet.CueIn],...
        [ThisNet.TrigAmp],...
        [TrigStart],...
        [trigEnd],...
        innateTotT);
    NoiseIn = ThisNet.generateNoiseInput(InPulses,ThisNet.innateNoiseLvl);
    ThisNet.randStateRRN;
    ThisNet.RNNStateGPU;
    ThisNet.trainRNNTargetGPU(thisTarg,...
        [trigEnd:ThisNet.trainTimeStep:innateTotT], NoiseIn);
    fprintf([ThisNet.getName,' TrainTrial:%i/%i Stim:%i/%i\n'],...
        trial,TrainTrials,stim,numel(unique(ThisNet.TrainStimOrder)))
end

ThisNet.reconWs; % reconstruct weights from GPU values
ThisNet.clearStateVars;
ThisNet.saveRNN(SaveDir);
%% train output
numOutTrials = 10;
XDurr=ThisNet.TargLen/2000;
AllTargTimes = [163,513,750,1200,1750]*XDurr+trigEnd;
OutDur = ThisNet.TargLen+trigEnd;
ThisNet.generateP_CPU;
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
    InPulses = ThisNet.generateInputPulses(...
        [2],...
        [ThisNet.TrigAmp],...
        [TrigStart],...
        [trigEnd],...
        ThisTotT);
    figure(inFig); clf; plot(InPulses');
    NoiseIn = ThisNet.generateNoiseInput(InPulses, ...
        ThisNet.innateNoiseLvl);
    hEx = zeros(ThisNet.numEx, ThisTotT);
    hOut = zeros(ThisNet.numOut, ThisTotT);
    ThisNet.randStateRRN;
    for t = 1:ThisTotT
        In = NoiseIn(:,t);
        [~, hEx(:,t)] = ThisNet.IterateRNN_CPU(In);
        hOut(:,t) = ThisNet.IterateOutCPU;
        if ismember(t,outTrnWind)
            ThisNet.trainOutputFORCE(OutTarget(:,t));
        end
    end
    figure(recFig); plot(hEx(10,:));
    figure(outFig); plot(hOut'); drawnow;
end
ThisNet.clearStateVars;
ThisNet.saveRNN(SaveDir)
%% Test output
ThisNet.SpeedIn=1;
TestNovelCue(ThisNet,SaveDir,1,1,1,10)