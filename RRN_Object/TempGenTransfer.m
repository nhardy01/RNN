% Seed = 69;%,96,6969,777,42];
% targLen = 2000;
% numTargs=2;
% trigAmp = 5;
% trigStart = 500;
% restDur = 30000;
% postTime = 200;
% N = 1800;
% numOutTrials = 20;
% trainStep = 5;
% NumOut = 2;
% NumIn = 3;
% noiseL = 0.01;
% scaleDir = 1;
% CueOrder=[1,2,2];
% TonicOrder=[1,1,2];
% %% Testing Parameters
% trainTrials = 90;
% ScaleFactor = [3]; % 2/3
% Tau = [50]; % 10
% G = [1.6]; % 1.6
% SpeedSigs=[0.075,0.3];
% speedTic=diff(SpeedSigs);
% if scaleDir>0
%     tonicBL = max(SpeedSigs); % 0.4
% else
%     tonicBL = min(SpeedSigs); % 0.4
% end
% 
% %% Run
% MasterSaveDir = '~/Documents/Nick/TemporalInvariance/GenTransfer/';
% 
% InFig = figure; NoiseFig = figure; TargFig = figure;
% %% Set up save folder
% SaveDir = fullfile(MasterSaveDir,'OneSpeedIn');
% if ~exist(SaveDir)
%     mkdir(SaveDir)
% end
% fprintf([SaveDir, '\n'])
% ThisNet = RNN(Seed, G, N, NumIn, NumOut); %initiate network
% %% Set standard parameters
% ThisNet.TrigAmp = trigAmp;
% ThisNet.TrigDur = Tau*5;
% ThisNet.TargLen = targLen;
% ThisNet.innateNoiseLvl = noiseL;
% ThisNet.scaleDir = scaleDir;
% %% Set testing parameters
% ThisNet.scalingFactor = ScaleFactor;
% ThisNet.tau = Tau;
% ThisNet.originalTonicLvl = tonicBL;
% ThisNet.scalingTics = speedTic;
% ThisNet.ExExTrainTonicStims =...
%     [ThisNet.originalTonicLvl,...
%     ThisNet.originalTonicLvl-(ThisNet.scalingTics*ThisNet.scaleDir)];
% trigEnd = ThisNet.TrigDur + trigStart;
% innateTotT = ThisNet.TargLen + trigEnd + restDur;
% %% Generate the RNN target
% AllTargs={};
% for targInd=1:numTargs
%     InPulses = ThisNet.generateInputPulses(...
%         [targInd, 3], [ThisNet.TrigAmp, ThisNet.originalTonicLvl],...
%         [trigStart, trigStart],...
%         [trigEnd, trigEnd+ThisNet.TargLen], innateTotT);
%     NoiseIn = ThisNet.generateNoiseInput(InPulses, ...
%         0);
%     InnateTarg = zeros(ThisNet.numEx, innateTotT);
%     ThisNet.randStateRRN;
%     for t = 1:innateTotT
%         In = NoiseIn(:,t);
%         [~, InnateTarg(:,t)] = ThisNet.IterateRNN_CPU(In);
%     end
%     AllTargs{targInd} = ThisNet.gatedExTarget(InnateTarg,...
%         innateTotT-restDur, 30);
% end
% [ScaledTargs, ~] = ThisNet.scaleTarget(AllTargs{2},...
%     trigEnd, trigEnd+ThisNet.TargLen);
% AllTargs{2}=ScaledTargs{1};AllTargs{3}=ScaledTargs{2}; clear ScaledTargs;
% TargTimes=[size(AllTargs{1},2),size(AllTargs{2},2),size(AllTargs{3},2)];
% %% Train RNN
% ThisNet.setRNNTarget(AllTargs);
% ThisNet.generate_W_P_GPU;
% tic
% for trial = 1:trainTrials
%     %tic
%     stim = mod(trial-1,numel(AllTargs))+1;
%     thisTarg = gpuArray(single(AllTargs{stim}));
%     thisTrialTrainTime = TargTimes(stim);
%     sigEnd = thisTrialTrainTime - restDur;
%     figure(TargFig); clf; imagesc(thisTarg); title('Target');
%     thisSpeedSig = ThisNet.ExExTrainTonicStims(TonicOrder(stim));
%     InPulses = ThisNet.generateInputPulses(...
%         [CueOrder(stim), 3], [ThisNet.TrigAmp, thisSpeedSig],...
%         [trigStart, trigStart],...
%         [trigEnd, sigEnd], thisTrialTrainTime);
%     figure(InFig);clf; plot(InPulses'); title('Input')
%     NoiseIn = ThisNet.generateNoiseInput(InPulses, ...
%         ThisNet.innateNoiseLvl);
%     figure(NoiseFig);clf; plot(NoiseIn'); title('InPlusNoise');
%     ThisNet.randStateRRN;
%     ThisNet.RNNStateGPU;
%     ThisNet.trainRNNTargetGPU(thisTarg,...
%         [trigEnd:trainStep:thisTrialTrainTime], NoiseIn);
%     %figure(TrainFig);clf; imagesc(hTrain); title('TrainingHistory');
%     drawnow;
%     %clear hTrain
%     %toc
% end
% ThisNet.reconWs; % reconstruct weights from GPU values
% ThisNet.clearStateVars;
% ThisNet.saveRNN(SaveDir);
%% train output
OutTrainStim = [1,2];
OutDur = TargTimes(OutTrainStim)-restDur+200;
OutTotT = OutDur + 200;
ThisNet.generateP_CPU;
AllTargTimes = 1750+trigEnd;
ThisNet.newWExOut;
ThisNet.newState(1);
for outStim=1:numel(OutTrainStim)
    OutTarget = zeros(ThisNet.numOut,OutDur(outStim));
    thisTotT=OutTotT(OutTrainStim(outStim));
    for targTInd = 1:numel(AllTargTimes)
        thisHitTime=AllTargTimes(targTInd);
        ThisHit = normpdf(1:OutDur,thisHitTime,50);
        ThisHit=(1/max(ThisHit)).*ThisHit;
        OutTarget(outStim,:) = OutTarget(outStim,:)+ThisHit;
    end
    OutTarget(outStim,:)=OutTarget(outStim,:)-mean(OutTarget);
    thisSpeedSig=ThisNet.ExExTrainTonicStims(OutTrainStim(outStim));
    outTrnWind = trigEnd:OutDur;
    InPulses = ThisNet.generateInputPulses(...
        [CueOrder(OutTrainStim(outStim)), 3], [ThisNet.TrigAmp,...
        ThisNet.ExExTrainTonicStims(OutTrainStim(outStim))],...
        [trigStart, trigStart],...
        [trigEnd, trigEnd+ThisNet.TargLen], thisTotT);
    outFig = figure; hold on; title('Out Train')
    plot(OutTarget','--k','linewidth',2);
    recFig = figure; hold on; title('RNNUnit Out Train')
    plot(AllTargs{OutTrainStim(outStim)}(10,:), '--k', 'linewidth', 2);
    for trialInd = 1:numOutTrials
        NoiseIn = ThisNet.generateNoiseInput(InPulses, ...
            ThisNet.innateNoiseLvl);
        hEx = zeros(ThisNet.numEx, thisTotT);
        hOut = zeros(ThisNet.numOut, thisTotT);
        ThisNet.randStateRRN;
        for t = 1:thisTotT
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
end
%% Test output
InterpSS = [min(ThisNet.ExExTrainTonicStims):...
    ThisNet.scalingTics/4:...
    max(ThisNet.ExExTrainTonicStims)];
for outStim=1:numel(OutTrainStim)
    thisOut=OutTrainStim(outStim);
    outFigT = figure; hold on;
    o1h = subplot(size(InterpSS,2),1,1); title(o1h,'Out Test');
    recFigT = figure; hold on;
    r1h = subplot(size(InterpSS,2),1,1);  title(r1h,'RNN Test');
    testOutTotT = ThisNet.TargLen*4+1000+trigEnd;
    InPulses = {};
    for trialInd = 1:numel(InterpSS)*5
        stim = mod(trialInd-1,numel(InterpSS))+1;
        thisSS = InterpSS(stim);
        numScales = (thisSS-ThisNet.originalTonicLvl)/...
            ThisNet.scalingTics*ThisNet.scaleDir;
        sigDur = round(ThisNet.TargLen*(1-numScales*ThisNet.scalingFactor));
        InPulse = ThisNet.generateInputPulses([thisOut, 3], [ThisNet.TrigAmp, thisSS],...
            [trigStart, trigStart], [trigEnd, trigEnd+sigDur] , testOutTotT);
        InPulses{trialInd} = InPulse;
        hEx = zeros(ThisNet.numEx, testOutTotT);
        hOut = zeros(ThisNet.numOut, testOutTotT);
        hIn = zeros(ThisNet.numIn, testOutTotT);
        ThisNet.randStateRRN;
        for t = 1:testOutTotT
            In = InPulse(:,t);
            hIn(:,t) = In;
            InNoise = ThisNet.getWInEx*In+randn(ThisNet.numEx,1)*ThisNet.innateNoiseLvl;
            [~, hEx(:,t)] = ThisNet.IterateRNN_CPU(InNoise);
            hOut(:,t) = ThisNet.IterateOutCPU;
        end
        figure(outFigT); subplot(size(InterpSS,2),1,stim);
        plot(hOut(1,:),'g');hold on;plot(hOut(2,:),'b');
        ylim([-.5 1]); plot(hIn'); drawnow;
        figure(recFigT); subplot(size(InterpSS,2),1,stim);
        plot(hEx(50,:)); hold on; drawnow;
    end
end
ThisNet.clearStateVars;
ThisNet.saveRNN(SaveDir);
toc
%clear ThisNet; gpuDevice(); close all;
