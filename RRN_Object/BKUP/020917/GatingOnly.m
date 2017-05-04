close all
clear
InFig = figure; NoiseFig = figure; TargFig = figure; TrainFig = figure;
%% Standard Parameters
Seed = 110;
targLen = 1000;
TrigAmp = 2;
trigStart = 100;
restDur = 3000;
N = 700;
trainTrials = 5;
Tau = 10;
G = 1.6;
TrigDur = Tau*5;
TrigEnd = trigStart + TrigDur;
innateTotT = targLen + TrigEnd + restDur;
InitTargWindow = [TrigEnd+1:TrigEnd+targLen+1];
%% Run
MasterSaveDir = '~/Documents/Data/TemporalInvariance/GatingOnly/';
ThisTau = Tau;
ThisG = G;
%% Set up save folder
SaveDir = fullfile(MasterSaveDir);
if ~exist(SaveDir)
    mkdir(SaveDir)
end
ThisNet = RNN(Seed); %initiate network
%% Set standard parameters
ThisNet.numEx = N;
ThisNet.TargLen = targLen;
ThisNet.tau = ThisTau;
ThisNet.setG(ThisG);
ThisNet.TrigDur = TrigDur;
ThisNet.TrigAmp = TrigAmp;

%% Generate the RNN target
InPulses = ThisNet.generateInputPulses(...
    [2],...
    ThisNet.TrigAmp,...
    trigStart,...
    TrigEnd,...
    innateTotT);
NoiseIn = ThisNet.generateNoiseInput(InPulses, ...
    ThisNet.innateNoiseLvl);
InnateTarg = zeros(ThisNet.numEx, innateTotT);
for t = 1:innateTotT
    In = NoiseIn(:,t);
    [~, InnateTarg(:,t)] = ThisNet.IterateRNN_CPU(In);
end

GatedTarg = ThisNet.gatedExTarget(InnateTarg,...
    innateTotT-restDur, 10);

%% Train
ThisNet.setRNNTarget(GatedTarg);
ThisNet.generate_W_P_GPU;
for trial = 1:trainTrials
    %tic
    figure(TargFig); clf; imagesc(GatedTarg); title('Target');
    InPulses = ThisNet.generateInputPulses(...
        2,...
        ThisNet.TrigAmp,...
        trigStart,...
        TrigEnd,...
        innateTotT);
    figure(InFig);clf; plot(InPulses'); title('Input')
    NoiseIn = ThisNet.generateNoiseInput(InPulses, ...
        ThisNet.innateNoiseLvl);
    figure(NoiseFig);clf; plot(NoiseIn'); title('InPlusNoise');
    ThisNet.randStateRRN;
    ThisNet.trainRNNTargetGPU(GatedTarg,...
        [TrigEnd+1:innateTotT], NoiseIn);
    %figure(TrainFig);clf; imagesc(hTrain); title('TrainingHistory');
    drawnow;
    %clear hTrain
    %toc
end
ThisNet.reconWs; % reconstruct weights from GPU values
ThisNet.saveRNN(SaveDir);