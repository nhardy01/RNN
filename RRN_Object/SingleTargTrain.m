clear; close all;
Seed = [1,7,99,96,6969,777,42,69,55,100];
targLen = 2000;
trigAmp = 5;
trigStart = 500;
restDur = 30000;
postTime = 200;
N = 1800;
numOutTrials = 10;
trainStep = 5;
NumOut = 1;
NumIn = 3;
noiseL = 0.25;
scaleDir = 1;
trainTrials = 30;
ScaleFactor = [3]; % 2/3
Tau = [50]; % 10
G = [1.6]; % 1.6
SpeedSigs=[0.075,0.3];
speedTic=diff(SpeedSigs);
if scaleDir>0
    tonicBL = max(SpeedSigs); % 0.4
else
    tonicBL = min(SpeedSigs); % 0.4
end
MasterSaveDir = '~/Documents/Nick/TemporalInvariance/GeneralizationMech/';

Seed69RNN.setWExEx(Seed69RNN.getWExExInit)
Seed69RNN.setRNNTarget(target{1})
Seed69RNN.ExExTrainTonicStims = Seed69RNN.ExExTrainTonicStims(1);
trigEnd = Seed69RNN.TrigDur + trigStart;
innateTotT = Seed69RNN.TargLen + trigEnd + restDur;
shortTarg=Seed69RNN.getRNNTarget;
times=size(shortTarg,2);
sigEnd = times-restDur;
InPulses = Seed69RNN.generateInputPulses(...
    [2, 3], [Seed69RNN.TrigAmp, Seed69RNN.ExExTrainTonicStims],...
    [trigStart, trigStart],...
    [trigEnd, sigEnd], times);
figure; plot(InPulses'); title('Input')
%% Train RNN
Seed69RNN.newState(666);
Seed69RNN.generate_W_P_GPU;
NoiseFig=figure; hold on;
for trial = 1:trainTrials
    NoiseIn = Seed69RNN.generateNoiseInput(InPulses, ...
        Seed69RNN.innateNoiseLvl);
    figure(NoiseFig);clf; plot(NoiseIn'); title('InPlusNoise');
    Seed69RNN.randStateRRN;
    Seed69RNN.RNNStateGPU;
    Seed69RNN.trainRNNTargetGPU(shortTarg,...
        [trigEnd:trainStep:times], NoiseIn);
    drawnow;
end
Seed69RNN.reconWs; % reconstruct weights from GPU values
Seed69RNN.clearStateVars;
%% train output
OutTrainStim = 1;
OutDur = times(OutTrainStim)-restDur+200;
OutTotT = OutDur + 200;
Seed69RNN.generateP_CPU;
AllTargTimes = [163,513,750,1200,1750]+trigEnd;
OutTarget = zeros(1,OutDur);
for targTInd = 1:numel(AllTargTimes)
    thisHitTime=AllTargTimes(targTInd);
    ThisHit = normpdf(1:OutDur,thisHitTime,50);
    ThisHit=(1/max(ThisHit)).*ThisHit;
    OutTarget = OutTarget+ThisHit;
    %OutTarget(AllTargTimes(targTInd)-45:AllTargTimes(targTInd)+45)=1;
end
%OutTarget=(2/max(OutTarget)).*OutTarget;
OutTarget=OutTarget-mean(OutTarget);
outTrnWind = trigEnd:OutDur;
Seed69RNN.newState(1);
InPulses = Seed69RNN.generateInputPulses(...
    [2, 3], [Seed69RNN.TrigAmp,...
    Seed69RNN.ExExTrainTonicStims(OutTrainStim)],...
    [trigStart, trigStart],...
    [trigEnd, trigEnd+Seed69RNN.TargLen], OutTotT);
outFig = figure; hold on; title('Out Train')
plot(OutTarget,'--k','linewidth',2);
recFig = figure; hold on; title('RNNUnit Out Train')
plot(ScaledTargs{OutTrainStim}(10,:), '--k', 'linewidth', 2);
for trialInd = 1:numOutTrials
    NoiseIn = Seed69RNN.generateNoiseInput(InPulses, ...
        Seed69RNN.innateNoiseLvl);
    hEx = zeros(Seed69RNN.numEx, OutTotT);
    hOut = zeros(Seed69RNN.numOut, OutTotT);
    Seed69RNN.randStateRRN;
    for t = 1:OutTotT
        In = NoiseIn(:,t);
        [~, hEx(:,t)] = Seed69RNN.IterateRNN_CPU(In);
        hOut(:,t) = Seed69RNN.IterateOutCPU;
        if ismember(t,outTrnWind)
            Seed69RNN.trainOutputFORCE(OutTarget(:,t));
        end
    end
    figure(recFig); plot(hEx(10,:));drawnow;
    figure(outFig); plot(hOut'); drawnow;
end
%% Test output
InterpSS = [min(Seed69RNN.ExExTrainTonicStims):...
    Seed69RNN.scalingTics/4:...
    max(Seed69RNN.ExExTrainTonicStims)];
outFigT = figure; hold on;
o1h = subplot(size(InterpSS,2),1,1); title(o1h,'Out Test');
recFigT = figure; hold on;
r1h = subplot(size(InterpSS,2),1,1);  title(r1h,'RNN Test');
testOutTotT = Seed69RNN.TargLen*4+1000+trigEnd;
InPulses = {};
for trialInd = 1:numel(InterpSS)*5
    stim = mod(trialInd-1,numel(InterpSS))+1;
    thisSS = InterpSS(stim);
    numScales = (thisSS-Seed69RNN.originalTonicLvl)/...
        Seed69RNN.scalingTics*Seed69RNN.scaleDir;
    sigDur = round(Seed69RNN.TargLen*(1-numScales*Seed69RNN.scalingFactor));
    %sigDur=29880*exp(thisSS*-10.57)+200;
    InPulse = Seed69RNN.generateInputPulses([2, 3], [Seed69RNN.TrigAmp, thisSS],...
        [trigStart, trigStart], [trigEnd, trigEnd+sigDur] , testOutTotT);
    InPulses{trialInd} = InPulse;
    %InPlusNoise = Seed69RNN.generateNoiseInput(InPulse, Seed69RNN.innateNoiseLvl);
    hEx = zeros(Seed69RNN.numEx, testOutTotT);
    hOut = zeros(Seed69RNN.numOut, testOutTotT);
    hIn = zeros(Seed69RNN.numIn, testOutTotT);
    Seed69RNN.randStateRRN;
    for t = 1:testOutTotT
        In = InPulse(:,t);
        hIn(:,t) = In;
        InNoise = Seed69RNN.getWInEx*In+randn(Seed69RNN.numEx,1)*Seed69RNN.innateNoiseLvl;
        [~, hEx(:,t)] = Seed69RNN.IterateRNN_CPU(InNoise);
        hOut(:,t) = Seed69RNN.IterateOutCPU;
    end
    figure(outFigT); subplot(size(InterpSS,2),1,stim);
    plot(hOut'); ylim([-.5 1]); hold on; plot(hIn'); drawnow;
    figure(recFigT); subplot(size(InterpSS,2),1,stim);
    plot(hEx(50,:)); hold on; drawnow;
end
Seed69RNN.clearStateVars;
Seed69RNN.saveRNN(SaveDir);

