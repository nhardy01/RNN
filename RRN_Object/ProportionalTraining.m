clear; close all;
Seed = [1,7,99,96,6969,777,42,69,55,100];
targLen = 2000;
numTargs=1;
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
PropMajor=1; % Proportion to train major stim. Ratio= ProbMajor:1(stim 1):1(stim 2 if applicable)
MajorStim=1; % Stim to train the most often. Innate stim is independent of this. 1= fastest, 2=middle, 3=slowest
trainTrials=30;
ScaleFactor = [3]; % 2/3
Tau = [50]; % 10
G = [1.6]; % 1.6
ExtreemSpeedSigs=[0.15];
InnateSpeed=1;
TrainSpeeds=[1];
NumTics=1;
speedTic=abs(diff(ExtreemSpeedSigs));
% tonicBL=max(ExtreemSpeedSigs)-speedTic/2;
%{
Try doing middle as baseline speed, and training middle speed most often
but keeping BL speed as fastest
%}
% if scaleDir>0
%     tonicBL = max(ExtreemSpeedSigs); % 0.4
% else
%     tonicBL = min(ExtreemSpeedSigs); % 0.4
% end
MasterSaveDir = '~/Documents/Nick/TemporalInvariance/ProportionalTraining/';
%% Run
for seedInd=1:numel(Seed)
    thisSeed=Seed(seedInd);
    for blInd=1:numel(MajorStim)
        thisMajorStim=MajorStim(blInd);
        for propInd=1:numel(PropMajor)
            thisPropMajor=PropMajor(propInd);
            for trlInd=1:numel(trainTrials)
                thisTrnTrial=trainTrials(trlInd);
                % FOR INTERP FIRST TRAINING
                %StimOrder=ones(1,trainTrials)*thisMajorStim;
                %offTics=1:NumTics;
                %offTics(thisMajorStim)=[];
                %transitionPt=(trainTrials/(thisPropMajor+numel(offTics)))*numel(offTics);
                % FOR FULL (SHORT,MID,LONG) TRAIN FIRST
                StimOrder=ones(1,trainTrials)*thisMajorStim;
                offTics=1:NumTics;
                offTics(thisMajorStim)=[];
                transitionPt=(trainTrials/(thisPropMajor+numel(offTics)))*NumTics;
                %%%%
                for i=1:numel(offTics)
                    %StimOrder(thisPropBL+1:(thisPropBL+1)*i:end)=offTics(i);
                    % CHANGED TO TRAIN FULL FIRST, THEN 1X
                    StimOrder(offTics(i):NumTics:transitionPt)=offTics(i);
                end
                InFig = figure; NoiseFig = figure; TargFig = figure;
                %% Set up save folder
%                 SaveDir = fullfile(MasterSaveDir,...
%                     sprintf('MajorStim%i',thisMajorStim),...
%                     sprintf('%ixMiddleToExtreme',thisPropMajor),...
%                     sprintf('%iTrainTrials',thisTrnTrial));
                SaveDir = fullfile(MasterSaveDir,...
                    'MiddleSpeedOnly',...
                    sprintf('TrainNoise_%.2g',noiseL),...
                    sprintf('%iTrainTrials',thisTrnTrial));
                if ~exist(SaveDir,'dir')
                    mkdir(SaveDir)
                end
                fprintf([SaveDir, '\n'])
                ThisNet = RNN(thisSeed, G, N, NumIn, NumOut); %initiate network
                %% Set standard parameters
                ThisNet.TrigAmp = trigAmp;
                ThisNet.TrigDur = Tau*5;
                ThisNet.TargLen = targLen;
                ThisNet.innateNoiseLvl = noiseL;
                ThisNet.scaleDir = scaleDir;
                %% Set testing parameters
                ThisNet.TrainStimOrder = StimOrder;
                ThisNet.scalingFactor = ScaleFactor;
                ThisNet.tau = Tau;
                %ThisNet.originalTonicLvl = tonicBL;
                ThisNet.scalingTics = speedTic;
                ThisNet.ExExTrainTonicStims =...
                    (TrainSpeeds.^-1)*max(ExtreemSpeedSigs);
                ThisNet.originalTonicLvl=ThisNet.ExExTrainTonicStims(InnateSpeed);
                trigEnd = ThisNet.TrigDur + trigStart;
                innateTotT = ThisNet.TargLen + trigEnd + restDur;
                %% Generate the RNN target
                InPulses = ThisNet.generateInputPulses(...
                    [2, 3], [ThisNet.TrigAmp, ThisNet.originalTonicLvl],...
                    [trigStart, trigStart],...
                    [trigEnd, trigEnd+ThisNet.TargLen], innateTotT);
                NoiseIn = ThisNet.generateNoiseInput(InPulses, ...
                    0);
                InnateTarg = zeros(ThisNet.numEx, innateTotT);
                ThisNet.randStateRRN;
                for t = 1:innateTotT
                    In = NoiseIn(:,t);
                    [~, InnateTarg(:,t)] = ThisNet.IterateRNN_CPU(In);
                end
                GatedTarg = ThisNet.gatedExTarget(InnateTarg,...
                    innateTotT-restDur, 30);
                
                [ScaledTargs, TargTimes] = ThisNet.scaleTarget(GatedTarg,...
                    trigEnd, trigEnd+ThisNet.TargLen);
                clear GatedTarg InnateTarg;
                %% Train RNN
                ThisNet.setRNNTarget(ScaledTargs);
                ThisNet.generate_W_P_GPU;
                tic
                for trial = 1:thisTrnTrial
                    %tic
                    stim = ThisNet.TrainStimOrder(trial);
                    thisTarg = gpuArray(single(ScaledTargs{stim}));
                    thisTrialTrainTime = TargTimes(stim);
                    sigEnd = thisTrialTrainTime - restDur;
                    figure(TargFig); clf; imagesc(thisTarg); title('Target');
                    thisSpeedSig = ThisNet.ExExTrainTonicStims(stim);
                    InPulses = ThisNet.generateInputPulses(...
                        [2, 3], [ThisNet.TrigAmp, thisSpeedSig],...
                        [trigStart, trigStart],...
                        [trigEnd, sigEnd], thisTrialTrainTime);
                    figure(InFig);clf; plot(InPulses'); title('Input')
                    NoiseIn = ThisNet.generateNoiseInput(InPulses, ...
                        ThisNet.innateNoiseLvl);
                    figure(NoiseFig);clf; plot(NoiseIn'); title('InPlusNoise');
                    ThisNet.randStateRRN;
                    ThisNet.RNNStateGPU;
                    ThisNet.trainRNNTargetGPU(thisTarg,...
                        [trigEnd:trainStep:thisTrialTrainTime], NoiseIn);
                    drawnow;
                end
                ThisNet.reconWs; % reconstruct weights from GPU values
                ThisNet.clearStateVars;
                ThisNet.saveRNN(SaveDir);
                %% train output
                OutTrainStim = 1;%thisMajorStim;
                OutDur = TargTimes(OutTrainStim)-restDur+200;
                OutTotT = OutDur + 200;
                ThisNet.generateP_CPU;
                AllTargTimes = ([163,513,750,1200,1750])*TrainSpeeds(OutTrainStim)+trigEnd;
                OutTarget = zeros(ThisNet.numOut,OutDur);
                for targTInd = 1:numel(AllTargTimes)
                    thisHitTime=AllTargTimes(targTInd);
                    ThisHit = normpdf(1:OutDur,thisHitTime,50);
                    ThisHit=(1/max(ThisHit)).*ThisHit;
                    OutTarget = OutTarget+ThisHit;
                end
                %OutTarget=(2/max(OutTarget)).*OutTarget;
                OutTarget=OutTarget-mean(OutTarget);
                outTrnWind = trigEnd:OutDur;
                ThisNet.newState(54);
                InPulses = ThisNet.generateInputPulses(...
                    [2, 3], [ThisNet.TrigAmp,...
                    ThisNet.ExExTrainTonicStims(OutTrainStim)],...
                    [trigStart, trigStart],...
                    [trigEnd, trigEnd+ThisNet.TargLen], OutTotT);
                outFig = figure; hold on; title('Out Train')
                plot(OutTarget,'--k','linewidth',2);
                recFig = figure; hold on; title('RNNUnit Out Train')
                plot(ScaledTargs{OutTrainStim}(10,:), '--k', 'linewidth', 2);
                for trialInd = 1:numOutTrials
                    NoiseIn = ThisNet.generateNoiseInput(InPulses, ...
                        ThisNet.innateNoiseLvl);
                    hEx = zeros(ThisNet.numEx, OutTotT);
                    hOut = zeros(ThisNet.numOut, OutTotT);
                    ThisNet.randStateRRN;
                    for t = 1:OutTotT
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
                %% Test output
                %         InterpSS = [min(ThisNet.ExExTrainTonicStims):...
                %         ThisNet.scalingTics/4:...
                %         max(ThisNet.ExExTrainTonicStims)];
                %         outFigT = figure; hold on;
                %         o1h = subplot(size(InterpSS,2),1,1); title(o1h,'Out Test');
                %         recFigT = figure; hold on;
                %         r1h = subplot(size(InterpSS,2),1,1);  title(r1h,'RNN Test');
                %         testOutTotT = ThisNet.TargLen*4+1000+trigEnd;
                %         InPulses = {};
                %         for trialInd = 1:numel(InterpSS)*5
                %             stim = mod(trialInd-1,numel(InterpSS))+1;
                %             thisSS = InterpSS(stim);
                %             numScales = (thisSS-ThisNet.originalTonicLvl)/...
                %                 ThisNet.scalingTics*ThisNet.scaleDir;
                %             sigDur = round(ThisNet.TargLen*(1-numScales*ThisNet.scalingFactor));
                %             InPulse = ThisNet.generateInputPulses([2, 3], [ThisNet.TrigAmp, thisSS],...
                %                 [trigStart, trigStart], [trigEnd, trigEnd+sigDur] , testOutTotT);
                %             InPulses{trialInd} = InPulse;
                %             hEx = zeros(ThisNet.numEx, testOutTotT);
                %             hOut = zeros(ThisNet.numOut, testOutTotT);
                %             hIn = zeros(ThisNet.numIn, testOutTotT);
                %             ThisNet.randStateRRN;
                %             for t = 1:testOutTotT
                %                 In = InPulse(:,t);
                %                 hIn(:,t) = In;
                %                 InNoise = ThisNet.getWInEx*In+randn(ThisNet.numEx,1)*ThisNet.innateNoiseLvl;
                %                 [~, hEx(:,t)] = ThisNet.IterateRNN_CPU(InNoise);
                %                 hOut(:,t) = ThisNet.IterateOutCPU;
                %             end
                %             figure(outFigT); subplot(size(InterpSS,2),1,stim);
                %             plot(hOut(1,:),'g');hold on;
                %             ylim([-.5 1]); plot(hIn'); drawnow;
                %             figure(recFigT); subplot(size(InterpSS,2),1,stim);
                %             plot(hEx(50,:)); hold on; drawnow;
                %         end
            end
            ThisNet.clearStateVars;
            ThisNet.saveRNN(SaveDir);
            toc
            clear ThisNet; gpuDevice(); close all;
        end
    end
end
%clear ThisNet; gpuDevice(); close all;
