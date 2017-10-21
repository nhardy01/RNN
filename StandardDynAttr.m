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
Net = RNN(Seed, G, N, NumIn, NumOut); %initiate network
%% Set standard parameters
stimOrder = StimOrder;
aCue = TrigAmp;
Net.TrigDur = TrigDur;
Net.TargLen = TargLen;
aNoise = NoiseL;
Net.scaleDir=0;
%% Set testing parameters
Net.scalingFactor=0;
Net.tau=Tau;
Net.originalTonicLvl = 0;
Net.scalingTics = 0;
Net.ExExTrainTonicStims=0;
trigEnd = Net.TrigDur + trigStart;
innateTotT = Net.TargLen + trigEnd;
%% Generate the RNN target
InPulses = Net.generateInputPulses(...
    2,...
    ThisaCue,...
    trigStart,...
    trigEnd,...
    innateTotT);
NoiseIn = Net.generateNoiseInput(InPulses, ...
    0);
InnateTarg = zeros(Net.nRec, innateTotT);
Net.randStateRRN;
for t = 1:innateTotT
    In = NoiseIn(:,t);
    [~, InnateTarg(:,t)] = Net.IterateRNN_CPU(In);
end
%% Train RNN
Net.setRNNTarget(InnateTarg);
Net.generate_W_P_GPU;
tic
thisTarg = gpuArray(single(InnateTarg));
TargFig=figure; imagesc(InnateTarg); title('Target');
InFig=figure; plot(InPulses'); title('Input');
NoiseFig=figure;
for trial = 1:TrainTrials
    NoiseIn = Net.generateNoiseInput(InPulses, ...
        aNoise);
    figure(NoiseFig);clf; plot(NoiseIn'); title('InPlusNoise');
    Net.randStateRRN;
    Net.RNNStateGPU;
    Net.trainRNNFORCE_GPU(thisTarg,...
        [trigEnd:trainStep:innateTotT], NoiseIn);
    drawnow;
end
Net.reconWs; % reconstruct weights from GPU values
Net.clearStateVars;
%% train output
OutDur = Net.TargLen+trigEnd;
Net.generateP_CPU;
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
Net.newState(Net.getNetworkSeed*31);
outFig = figure; hold on; title('Out Train')
plot(OutTarget,'--k','linewidth',2);
recFig = figure; hold on; title('RNNUnit Out Train')
plot(InnateTarg(10,:), '--k', 'linewidth', 2);
for trialInd = 1:numOutTrials
    NoiseIn = Net.generateNoiseInput(InPulses, ...
        aNoise);
    hEx = zeros(Net.nRec, OutDur);
    hOut = zeros(Net.numOut, OutDur);
    Net.randStateRRN;
    for t = 1:OutDur
        In = NoiseIn(:,t);
        [~, hEx(:,t)] = Net.IterateRNN_CPU(In);
        hOut(:,t) = Net.IterateOutCPU;
        if ismember(t,outTrnWind)
            Net.trainOutFORCE(OutTarget(:,t));
        end
    end
    figure(recFig); plot(hEx(10,:));drawnow;
    figure(outFig); plot(hOut'); drawnow;
end
Net.clearStateVars;
Net.saveRNN(SaveDir);
clear Net; gpuDevice(); close all;
