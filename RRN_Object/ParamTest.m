close all; clear;
%% Standard Parameters
Seeds=100;%[7,69,99,96,6969,777,42,69,55,100];%1
targLen = 2000;
trigAmp=5;
trigStart=500;
restDur=30000;
postTime=200;
N=1800;
numOutTrials=10;
NumOut=1;
NumIn=3;
noiseL=0.05;
scaleDir=1;
%% Testing Parameters
trainTrials=60;
ScaleFactor=[3]; % 2/3
Tau=[50]; % 10
trainStep=Tau/10;
G=[1.6]; % 1.6
SpeedSigs=[0.075,0.3];
StimOrder=ones(1,trainTrials);
for i=2:numel(SpeedSigs); StimOrder(i:numel(SpeedSigs):end)=i; end;
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
                        for trnTrlInd = 1:numel(trainTrials);
                            tic
                            thisTrials = trainTrials(trnTrlInd);
                            %% Set up save folder
                            SaveDir = fullfile(MasterSaveDir,...
                                sprintf('Factor_%.2g',ThisScaleF),...
                                sprintf('Tau_%i',ThisTau),...
                                sprintf('G_%.3g',ThisG),...
                                sprintf('ScaleDir_%g',scaleDir),...
                                sprintf('SpeedBL_%.3g',ThisBL),...
                                sprintf('TrainTrial_%i',thisTrials),...
                                sprintf('NoiseAmp_%.2g',noiseL),...
                                sprintf('TargLen_%i',targLen)); 
                            if ~exist(SaveDir)
                                mkdir(SaveDir)
                            end
                            fprintf([SaveDir,'\n'])
                            ThisNet = RNN(Seed, ThisG, N, NumIn, NumOut); %initiate network
                            %% Set standard parameters
                            ThisNet.TrainStimOrder = StimOrder;
                            ThisNet.TrigAmp = trigAmp;
                            ThisNet.TrigDur = trigDur;
                            ThisNet.TargLen = targLen;
                            ThisNet.innateNoiseLvl = noiseL;
                            ThisNet.scaleDir = scaleDir;
                            ThisNet.scalingFactor = ThisScaleF;
                            ThisNet.tau = ThisTau;
                            ThisNet.originalTonicLvl = ThisBL;
                            ThisNet.ExExTrainTonicStims=SpeedSigs;
                            ThisNet.restDur=restDur;
                            ThisNet.trainTimeStep=trainStep;
                            %% Train RNN
                            TrainRNN_TempInv(ThisNet,thisTrials);
                            ThisNet.clearStateVars;
                            ThisNet.saveRNN(SaveDir);
                            %% train output
                            TrainOut(ThisNet,SaveDir);
                            ThisNet.clearStateVars;
                            ThisNet.saveRNN(SaveDir)
                            %% Test output
                            TestRNN(ThisNet,SaveDir,1,20,0,false)
                            ThisNet.clearStateVars;
                            ThisNet.saveRNN(SaveDir);
                            toc
                            clear ThisNet; gpuDevice(); close all;
                        end
                end
            end
        end
    end
end
