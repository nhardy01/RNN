close all
clear

%% Standard Parameters
Seeds = [69]%,55,1,7,99];%,96,6969,777,42];
targLen = 2000;
trigAmp = 5;
trigStart = 500;
restDur = 30000;
postTime = 200;
N = 1800;
numOutTrials = 20;
trainStep = 5;
NumOut = 1;
NumIn = 3;
noiseL = 0.5;
scaleDir = 1;
%% Testing Parameters
trainTrials = 100;
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

%% Run
MasterSaveDir = '~/Documents/Nick/TemporalInvariance/ParamTest/';
for SeedInd=1:numel(Seeds)
    Seed=Seeds(SeedInd);
    for ScaleInd = 1:numel(ScaleFactor)
        ThisScaleF = ScaleFactor(ScaleInd);
        for TauInd = 1:numel(Tau)
            ThisTau = Tau(TauInd);
            trigDur = ThisTau*5;
            for GInd = 1:numel(G)
                ThisG = G(GInd);
                for BLInd = 1:numel(tonicBL)
                    ThisBL = tonicBL(BLInd);
                    for SpeedTInd = 1:numel(speedTic)
                        for trnTrlInd = 1:numel(trainTrials);
                            InFig = figure; NoiseFig = figure; TargFig = figure;
                            thisTrials = trainTrials(trnTrlInd);
                            %% Set up save folder
                            ThisTic = speedTic(SpeedTInd);
                            SaveDir = fullfile(MasterSaveDir,...
                                sprintf('Factor_%.2g',ThisScaleF),...
                                sprintf('Tau_%i',ThisTau),...
                                sprintf('G_%.3g',ThisG),...
                                sprintf('ScaleDir_%g',scaleDir),...
                                sprintf('SpeedBL_%.3g',ThisBL),...
                                sprintf('SpeedTic_%.3g',ThisTic),...
                                sprintf('TrainTrial_%.1g',thisTrials),...
                                sprintf('NoiseAmp_%.2g',noiseL));
                            if ~exist(SaveDir)
                                mkdir(SaveDir)
                            end
                            fprintf([SaveDir, '\n'])
                            ThisNet = RNN(Seed, ThisG, N, NumIn, NumOut); %initiate network
                            %% Set standard parameters
                            ThisNet.TrigAmp = trigAmp;
                            ThisNet.TrigDur = trigDur;
                            ThisNet.TargLen = targLen;
                            ThisNet.innateNoiseLvl = noiseL;
                            ThisNet.scaleDir = scaleDir;
                            %% Set testing parameters
                            ThisNet.scalingFactor = ThisScaleF;
                            ThisNet.tau = ThisTau;
                            ThisNet.originalTonicLvl = ThisBL;
                            ThisNet.scalingTics = ThisTic;
                            ThisNet.ExExTrainTonicStims =...
                                [ThisBL,ThisBL-(ThisTic*ThisNet.scaleDir)]; % changed for reversed scaling
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
                            %
                            GatedTarg = ThisNet.gatedExTarget(InnateTarg,...
                                innateTotT-restDur, 30);
                            [ScaledTargs, times] = ThisNet.scaleTarget(GatedTarg,...
                                trigEnd, trigEnd+ThisNet.TargLen);
                            
                            %% Train RNN
                            ThisNet.setRNNTarget(ScaledTargs);
                            ThisNet.generate_W_P_GPU;
                            tic
                            for trial = 1:thisTrials
                                %tic
                                stim = mod(trial-1,numel(ThisNet.ExExTrainTonicStims))+1;
                                thisTrialTrainTime = times(stim);
                                sigEnd = times(stim) - restDur;
                                thisTarg = gpuArray(single(ScaledTargs{stim}));
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
                                %figure(TrainFig);clf; imagesc(hTrain); title('TrainingHistory');
                                drawnow;
                                %clear hTrain
                                %toc
                            end
                            ThisNet.reconWs; % reconstruct weights from GPU values
                            ThisNet.clearStateVars;
                            %% train output
                            OutTrainStim = 1;
                            OutDur = times(OutTrainStim)-restDur+200;
                            OutTotT = OutDur + 200;
                            ThisNet.generateP_CPU;
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
                            ThisNet.newState(1);
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
                            InterpSS = [min(ThisNet.ExExTrainTonicStims):...
                                ThisNet.scalingTics/4:...
                                max(ThisNet.ExExTrainTonicStims)];
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
                                %sigDur=29880*exp(thisSS*-10.57)+200;
                                InPulse = ThisNet.generateInputPulses([2, 3], [ThisNet.TrigAmp, thisSS],...
                                    [trigStart, trigStart], [trigEnd, trigEnd+sigDur] , testOutTotT);
                                InPulses{trialInd} = InPulse;
                                %InPlusNoise = ThisNet.generateNoiseInput(InPulse, ThisNet.innateNoiseLvl);
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
                                plot(hOut'); ylim([-.5 1]); hold on; plot(hIn'); drawnow;
                                figure(recFigT); subplot(size(InterpSS,2),1,stim);
                                plot(hEx(50,:)); hold on; drawnow;
                            end
                            ThisNet.clearStateVars;
                            ThisNet.saveRNN(SaveDir);
                            toc
                            %clear ThisNet; gpuDevice(); close all;
                        end
                    end
                end
            end
        end
    end
end