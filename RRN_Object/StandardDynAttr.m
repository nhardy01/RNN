clear;close all;
MasterSaveDir='~/Documents/Nick/TemporalInvariance/StandardDynAttr';
Seed=69;
G=1.6;
N=1800;
NumIn=3;
NumOut=1;
TrigAmp=5;
Tau=50;
TargLen=2000;
TrigDur=Tau*5;
NoiseL=0.05;
TrainTrials=15;
trainStep=5;
trigStart=500;
numOutTrials=10;
StimOrder=ones(1,TrainTrials);
SaveDir = fullfile(MasterSaveDir,...
    sprintf('Tau_%i',Tau),...
    sprintf('G_%.3g',G),...
    sprintf('TrainTrial_%.1g',TrainTrials),...
    sprintf('NoiseAmp_%.2g',NoiseL));
if ~exist(SaveDir)
    mkdir(SaveDir)
end
fprintf([SaveDir, '\n'])
ThisNet = RNN(Seed, G, N, NumIn, NumOut); %initiate network
%% Set standard parameters
ThisNet.TrainStimOrder = StimOrder;
ThisNet.TrigAmp = TrigAmp;
ThisNet.TrigDur = TrigDur;
ThisNet.TargLen = TargLen;
ThisNet.innateNoiseLvl = NoiseL;
ThisNet.scaleDir=0;
%% Set testing parameters
ThisNet.scalingFactor=0;
ThisNet.tau=Tau;
ThisNet.originalTonicLvl = 0;
ThisNet.scalingTics = 0;
ThisNet.ExExTrainTonicStims=0;
trigEnd = ThisNet.TrigDur + trigStart;
innateTotT = ThisNet.TargLen + trigEnd;
%% Generate the RNN target
InPulses = ThisNet.generateInputPulses(...
    2,...
    ThisNet.TrigAmp,...
    trigStart,...
    trigEnd,...
    innateTotT);
NoiseIn = ThisNet.generateNoiseInput(InPulses, ...
    0);
InnateTarg = zeros(ThisNet.numEx, innateTotT);
ThisNet.randStateRRN;
for t = 1:innateTotT
    In = NoiseIn(:,t);
    [~, InnateTarg(:,t)] = ThisNet.IterateRNN_CPU(In);
end
%% Train RNN
ThisNet.setRNNTarget(InnateTarg);
ThisNet.generate_W_P_GPU;
tic
thisTarg = gpuArray(single(InnateTarg));
TargFig=figure; imagesc(InnateTarg); title('Target');
InFig=figure; plot(InPulses'); title('Input');
NoiseFig=figure;
for trial = 1:TrainTrials
    NoiseIn = ThisNet.generateNoiseInput(InPulses, ...
        ThisNet.innateNoiseLvl);
    figure(NoiseFig);clf; plot(NoiseIn'); title('InPlusNoise');
    ThisNet.randStateRRN;
    ThisNet.RNNStateGPU;
    ThisNet.trainRNNTargetGPU(thisTarg,...
        [trigEnd:trainStep:innateTotT], NoiseIn);
    drawnow;
end
ThisNet.reconWs; % reconstruct weights from GPU values
ThisNet.clearStateVars;
%% train output
OutDur = ThisNet.TargLen+trigEnd;
ThisNet.generateP_CPU;
AllTargTimes = [163,513,750,1200,1750]+trigEnd;
OutTarget = zeros(1,OutDur);
for targTInd = 1:numel(AllTargTimes)
    thisHitTime=AllTargTimes(targTInd);
    ThisHit = normpdf(1:OutDur,thisHitTime,50);
    ThisHit=(1/max(ThisHit)).*ThisHit;
    OutTarget = OutTarget+ThisHit;
end
OutTarget=OutTarget-mean(OutTarget);
outTrnWind = trigEnd:OutDur;
ThisNet.newState(ThisNet.getNetworkSeed*31);
outFig = figure; hold on; title('Out Train')
plot(OutTarget,'--k','linewidth',2);
recFig = figure; hold on; title('RNNUnit Out Train')
plot(InnateTarg(10,:), '--k', 'linewidth', 2);
for trialInd = 1:numOutTrials
    NoiseIn = ThisNet.generateNoiseInput(InPulses, ...
        ThisNet.innateNoiseLvl);
    hEx = zeros(ThisNet.numEx, OutDur);
    hOut = zeros(ThisNet.numOut, OutDur);
    ThisNet.randStateRRN;
    for t = 1:OutDur
        In = NoiseIn(:,t);
        [~, hEx(:,t)] = ThisNet.IterateRNN_CPU(In);
        hOut(:,t) = ThisNet.IterateOutCPU;
        if ismember(t,outTrnWind)
            ThisNet.trainOutputFORCE(OutTarget(:,t));
        end
    end
    figure(recFig); plot(hEx(10,:));drawnow;
    figure(outFig); plot(hOut'); drawnow;
end
ThisNet.clearStateVars;
ThisNet.saveRNN(SaveDir);
clear ThisNet; gpuDevice(); close all;