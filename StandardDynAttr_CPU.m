%% Preamble
clearvars;
close all;
dirSaveBase = 'C:\Users\Nick\Documents\Box Sync\ResearchProjects\RNN_Master\Test';

%% Switches
bPlotRNNTrain   = false;
bPlotOutTrain   = true;
bPlotTest       = true;

%% Parameters
Seed         = 69;
G            = 1.6;
N            = 200;
NumIn        = 3;
NumOut       = 1;
TrigAmp      = 5;
Tau          = 10;
TargLen      = 500;
ampNoise     = 0.05;
TrainTrials  = 15;
numOutTrials = 10;
trigStart    = 250;
ixUCue       = 2;
trainStep    = Tau/10;
TrigDur      = Tau*5;
trigEnd      = TrigDur + trigStart;
innateTotT   = TargLen + trigEnd;
tOutPeak     = [TargLen * 0.75] + trigEnd;
SaveDir = fullfile(dirSaveBase,...
    sprintf('Tau_%i',Tau),...
    sprintf('G_%.3g',G),...
    sprintf('TrainTrial_%.1g',TrainTrials),...
    sprintf('NoiseAmp_%.2g',ampNoise));
if ~exist(SaveDir)
    mkdir(SaveDir)
end
fprintf('Save directory: %s\n',SaveDir)

%% Instantiate RNN
Net     = RNN(Seed, G, N, NumIn, NumOut); %initiate network
Net.tau = Tau;
%% Generate the RNN target
InPulses = Net.generateInputPulses(ixUCue, TrigAmp, trigStart, trigEnd,innateTotT);
NoiseIn = Net.generateNoiseInput(InPulses, 0);
InnateTarg = zeros(Net.nRec, innateTotT);
Net.randStateRRN;
for t = 1:innateTotT
    In = NoiseIn(:,t);
    [~, InnateTarg(:,t)] = Net.IterateRNN_CPU(In);
end

%% Train RNN
Net.setRNNTarget(InnateTarg);
Net.generateP_CPU;
tic
figure; ahErr = imagesc(InnateTarg); title('Error'); colorbar;
for trial = 1:TrainTrials
    fprintf('RNN training trial %i\n',trial)
    NoiseIn = Net.generateNoiseInput(InPulses, ampNoise);
    Net.randStateRRN;
    dRec = Net.trainRNNFORCE_CPU(InnateTarg,[trigEnd:trainStep:innateTotT], NoiseIn);
    if bPlotRNNTrain %%% Plot
        ahErr.CData = InnateTarg - dRec;
        keyboard;
    end
end
% clear state variables
Net.clearStateVars;
%% train output
OutDur = TargLen+trigEnd;
Net.generateP_CPU;
OutTarget = zeros(1,OutDur);
for targTInd = 1:numel(tOutPeak)
    thisHitTime = tOutPeak(targTInd);
    ThisHit = normpdf(1:OutDur,thisHitTime,50);
    ThisHit = (1/max(ThisHit)).*ThisHit;
    OutTarget = OutTarget+ThisHit;
end
OutTarget = OutTarget-mean(OutTarget);
outTrnWind = trigEnd:OutDur;
Net.newState(Net.getNetworkSeed*31);
if bPlotOutTrain
    outFig = figure; hold on; title('Out Train')
    plot(OutTarget,'--k','linewidth',2);
    recFig = figure; hold on; title('RNNUnit Out Train')
    plot(InnateTarg(10,:), '--k', 'linewidth', 2);
end
for trialInd = 1:numOutTrials
    NoiseIn = Net.generateNoiseInput(InPulses, ampNoise);
    hEx = zeros(Net.nRec, OutDur);
    hOut = zeros(Net.nOut, OutDur);
    Net.randStateRRN;
    for t = 1:OutDur
        In = NoiseIn(:,t);
        [~, hEx(:,t)] = Net.IterateRNN_CPU(In);
        hOut(:,t) = Net.IterateOutCPU;
        if ismember(t,outTrnWind)
            Net.trainOutFORCE(OutTarget(:,t));
        end
    end
    if bPlotOutTrain
        figure(recFig); plot(hEx(10,:));
        figure(outFig); plot(hOut');
    end
end
Net.clearStateVars;
Net.saveRNN(SaveDir);

%% Test
%TestRNN(Net, SaveDir, ampNoise, 2, 0, false, ixUCue, 1, 0)
TestRNN(Net, 1500)








