close all
clear
MasterSaveDir = 'D:\Lab\TemporalInvariance\New_RNN\Test';
%% Standard Parameters
Seeds = [69,55,1,7,99];% seeds to loop through
targLen = 2000; % orginal target length, will be scaled
trigAmp = 5; % cue input amplitude
trigStart = 500; % cue start
restDur = 30000; % time to train gated target
N = 1800; % network size
numOutTrials = 20; % output training trials
trainStep = 5; % train RNN every trainStep time points
NumOut = 1; % number of output units
NumIn = 3; % number of input units
noiseL = 0.05; % noise amplitude to train network at
scaleDir = 1; % +1:higher speed signal=shorter duration;-1:higher speed signal=longerduration
%% Testing Parameters
trainTrials = 60; % # of training trials for RNN
ScaleFactor = [3]; % scale target duration ScaleFactor+1 x
Tau = [50]; % time constant of units
G = [1.6]; % gain of recurrent weights
SpeedSigs=[0.075,0.3]; % speed signals to trian network with
speedTic=diff(SpeedSigs);
if scaleDir>0 % shorter duration should always be baseline
    tonicBL = max(SpeedSigs); % 0.4
else
    tonicBL = min(SpeedSigs); % 0.4
end

%% Run
for SeedInd=1:numel(Seeds) % loop through seeds
    Seed=Seeds(SeedInd); % set the seed
    for ScaleInd = 1:numel(ScaleFactor) % loop through scale factors
        ThisScaleF = ScaleFactor(ScaleInd); % set this scale factor
        for TauInd = 1:numel(Tau) % loop through test tau's
            ThisTau = Tau(TauInd); % set the tau
            trigDur = ThisTau*5; % set the trigger duration to 5x the tau
            for GInd = 1:numel(G) % loop through test G's
                ThisG = G(GInd); % set this G
                for BLInd = 1:numel(tonicBL) % loop throu test baselines
                    ThisBL = tonicBL(BLInd); % set this baseline
                    for SpeedTInd = 1:numel(speedTic) % loop through test speed tics
                        for trnTrlInd = 1:numel(trainTrials) % loop through train trail #'s
                            InFig = figure; NoiseFig = figure; TargFig = figure; % set up figures for training feedback
                            thisTrials = trainTrials(trnTrlInd); % set the # of train trials
                            %% Set up save folder
                            ThisTic = speedTic(SpeedTInd); % set the speed tic
                            SaveDir = fullfile(MasterSaveDir,... % define a directory to save the network after training, based on current test parameters
                                sprintf('Factor_%.2g',ThisScaleF),...
                                sprintf('Tau_%i',ThisTau),...
                                sprintf('G_%.3g',ThisG),...
                                sprintf('ScaleDir_%g',scaleDir),...
                                sprintf('SpeedBL_%.3g',ThisBL),...
                                sprintf('SpeedTic_%.3g',ThisTic),...
                                sprintf('TrainTrial_%.1g',thisTrials),...
                                sprintf('NoiseAmp_%.2g',noiseL));
                            if ~exist(SaveDir) % make the save if it doesn't exist already
                                mkdir(SaveDir)
                            end
                            SaveDir
                            ThisNet = RNN(Seed, ThisG, N, NumIn, NumOut); %initiate network
                            %% Set parameters
                            %{
                            The test parameters will be stored in the RNN
                            object, to allow for easy reference of what
                            parameters were used to train and test each
                            network. Training and testing should access the
                            parameters stored in the RNN to ensure
                            consistency
                            %}
                            ThisNet.TrigAmp = trigAmp; % trigger amplitude
                            ThisNet.TrigDur = trigDur; % trigger duration
                            ThisNet.TargLen = targLen; % target length
                            ThisNet.innateNoiseLvl = noiseL; % noise amplitude
                            ThisNet.scaleDir = scaleDir; % scaling direction
                            ThisNet.scalingFactor = ThisScaleF; % scaling factor
                            ThisNet.tau = ThisTau; % network time constant
                            ThisNet.originalTonicLvl = ThisBL; % baseline speed signal
                            ThisNet.scalingTics = ThisTic; % scaling tics
                            ThisNet.ExExTrainTonicStims =...
                                [ThisBL,ThisBL-(ThisTic*ThisNet.scaleDir)]; % speed signals to train the RNN with
                            trigEnd = ThisNet.TrigDur + trigStart; % trigger end, not an essential parameter
                            innateTotT = ThisNet.TargLen + trigEnd + restDur; % total time to run the simulation
                            %% Generate the RNN target
                            InPulses = ThisNet.generateInputPulses(... % generates input pulses for the whole trial
                                [2, 3], [ThisNet.TrigAmp, ThisNet.originalTonicLvl],...
                                [trigStart, trigStart],...
                                [trigEnd, trigEnd+ThisNet.TargLen], innateTotT);
                            NoiseIn = ThisNet.generateNoiseInput(InPulses,0); % combine the input with stochastic noise
                            InnateTarg = zeros(ThisNet.numEx, innateTotT); % preallocate target storage
                            ThisNet.randStateRRN; % set the network to a random Ex and ExV
                            for t = 1:innateTotT % loop through the entire length of the target
                                In = NoiseIn(:,t); % get the Input + noise for this time point
                                [~, InnateTarg(:,t)] = ThisNet.IterateRNN_CPU(In); % iterate the network and store the target
                            end
                            GatedTarg = ThisNet.gatedExTarget(InnateTarg,...
                                innateTotT-restDur, 30); % this function generates a gated target at the time of the second input, with decar time constant set at the third input
                            [ScaledTargs, times] = ThisNet.scaleTarget(GatedTarg,... 
                                trigEnd, trigEnd+ThisNet.TargLen); % scales the target in input 1, within the window defined by inputs 2 and 3, according the the stored ThisNet.ExExTrainTonicStims varaibles
                            
                            %% Train RNN
                            ThisNet.setRNNTarget(ScaledTargs); % set the network's target
                            ThisNet.generateP_CPU; % generate the P matrices
                            for trial = 1:thisTrials % trail loop for  training
                                stim = mod(trial-1,numel(ThisNet.ExExTrainTonicStims))+1; % set the stim #
                                thisTrialTrainTime = times(stim); % get the train time of this target
                                sigEnd = times(stim) - restDur; % speed signal offset
                                thisTarg = ScaledTargs{stim}; % choose target
                                figure(TargFig); clf; imagesc(thisTarg); title('Target'); % plot target feedback
                                thisSpeedSig = ThisNet.ExExTrainTonicStims(stim); % get correct speed signal
                                InPulses = ThisNet.generateInputPulses(...
                                    [2, 3], [ThisNet.TrigAmp, thisSpeedSig],...
                                    [trigStart, trigStart],...
                                    [trigEnd, sigEnd], thisTrialTrainTime); % define forrce input pulses
                                figure(InFig);clf; plot(InPulses'); title('Input') % input pulse feedback
                                NoiseIn = ThisNet.generateNoiseInput(InPulses, ...
                                    ThisNet.innateNoiseLvl); % define noise
                                figure(NoiseFig);clf; plot(NoiseIn'); title('InPlusNoise'); % noise feedback
                                ThisNet.randStateRRN; % randomize network state
                                ThisNet.trainFORCE(thisTarg,...
                                    [trigEnd:trainStep:thisTrialTrainTime], NoiseIn); % train the RNN with FORCe algorith on the CPU
                                drawnow;
                            end
                            ThisNet.clearStateVars; % clears the P, Ex, ExV, and other state variable to save space
                            %% train output
                            OutTrainStim = 1; % only train output at the baseline innate target
                            OutDur = times(OutTrainStim)-restDur+200; % output target length
                            OutTotT = OutDur + 200; % output trial duration
                            ThisNet.generateP_CPU; % generate the p matrices for output training
                            AllTargTimes = [163,513,750,1200,1750]+trigEnd; % target peak times based on psychophysics
                            OutTarget = zeros(1,OutDur); % preallocate the output target
                            for targTInd = 1:numel(AllTargTimes) % loop to generate the peaks
                                thisHitTime=AllTargTimes(targTInd);
                                ThisHit = normpdf(1:OutDur,thisHitTime,50);
                                ThisHit=(1/max(ThisHit)).*ThisHit;
                                OutTarget = OutTarget+ThisHit;
                            end
                            OutTarget=OutTarget-mean(OutTarget); % normalize out target by mean
                            outTrnWind = trigEnd:OutDur; % define training window
                            ThisNet.newState(1); % set rng to a different state
                            InPulses = ThisNet.generateInputPulses(...
                                [2, 3], [ThisNet.TrigAmp,...
                                ThisNet.ExExTrainTonicStims(OutTrainStim)],...
                                [trigStart, trigStart],...
                                [trigEnd, trigEnd+ThisNet.TargLen], OutTotT); % define pulses
                            outFig = figure; hold on; title('Out Train') % output target feedback
                            plot(OutTarget,'--k','linewidth',2);
                            recFig = figure; hold on; title('RNNUnit Out Train')
                            plot(ScaledTargs{OutTrainStim}(10,:), '--k', 'linewidth', 2); % show sampel RNN unit target for to verify convergence
                            for trialInd = 1:numOutTrials % loop through output traiing trials
                                NoiseIn = ThisNet.generateNoiseInput(InPulses, ...
                                    ThisNet.innateNoiseLvl); % noise
                                hEx = zeros(ThisNet.numEx, OutTotT); % preallocate
                                hOut = zeros(ThisNet.numOut, OutTotT);
                                ThisNet.randStateRRN; % set network to random state
                                for t = 1:OutTotT % iterate over t
                                    In = NoiseIn(:,t); % input at t
                                    [~, hEx(:,t)] = ThisNet.IterateRNN_CPU(In); % iterate RNN
                                    hOut(:,t) = ThisNet.IterateOutCPU; % iterate output
                                    if ismember(t,outTrnWind) % train ouput if win training window
                                        ThisNet.trainOutputFORCE(OutTarget(:,t));
                                    end
                                end
                                figure(recFig); plot(hEx(10,:));drawnow; % plot feedback
                                figure(outFig); plot(hOut'); drawnow;
                            end
                            %% Test output
                            InterpSS = [min(ThisNet.ExExTrainTonicStims):...
                                ThisNet.scalingTics/4:...
                                max(ThisNet.ExExTrainTonicStims)]; % test network at generalized speeds
                            outFigT = figure; hold on; % plot feedback
                            o1h = subplot(size(InterpSS,2),1,1); title(o1h,'Out Test');
                            recFigT = figure; hold on;
                            r1h = subplot(size(InterpSS,2),1,1);  title(r1h,'RNN Test');
                            testOutTotT = ThisNet.TargLen*4+1000+trigEnd;
                            InPulses = {};
                            for trialInd = 1:numel(InterpSS)*5 % test
                                stim = mod(trialInd-1,numel(InterpSS))+1;
                                thisSS = InterpSS(stim);
                                numScales = (thisSS-ThisNet.originalTonicLvl)/...
                                    ThisNet.scalingTics*ThisNet.scaleDir;
                                sigDur = round(ThisNet.TargLen*(1-numScales*ThisNet.scalingFactor));
                                InPulse = ThisNet.generateInputPulses([2, 3], [ThisNet.TrigAmp, thisSS],...
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
                                plot(hOut'); ylim([-.5 1]); hold on; plot(hIn'); drawnow;
                                figure(recFigT); subplot(size(InterpSS,2),1,stim);
                                plot(hEx(50,:)); hold on; drawnow;
                            end
                            ThisNet.clearStateVars;
                            ThisNet.saveRNN(SaveDir);
                            toc
                        end
                    end
                end
            end
        end
    end
end