clear;
tic
seeds = [110];
restEx = 1; % true/false for adding a "rest" period for training a gated attractor
restTime = 500; %time to train netowrk to rest for after innate trajectoy training
% testExNums = [0,20,50,80,100,200,300,400,500,600];
numEx = 1800; %total number of Ex units
testExNum = 1800; %number of plastic Ex units at a time
% seeds = [110 123433 45 98 123]; %seed to use to generate weights
initStims = [2]; %stim level for initializeing RRN activity. uses WIn(stim)
ExExTrainTonicStims = [0.35 0.15 0.55];%stims level 2 for tonic RRN stim uses WIn(3)
ExOutTrainTonicStim = [0.35];
testTonicStims = [0.35 0.15 0.55 0.25 0.45 0.05 0.65];
rrnTrainTrials = 500; %number of iterations for training recurrent network
wExOutTrainTrials = length(ExOutTrainTonicStim)*15; %number of iterations for training WExOut
shortInnateTarget = 1; %boolean for downsampling the innate target
scaleDir = 1; %+/- 1 indicating whether an increase in tonic stim level scales the target up or down
postTime = 150;
testG = 1.6;
multipleTrigInputs = 0;
recOnVar = 1;
tau   = 10;
recNoiseLvl = 0.01;
WExOutNoiseLvl = 0.01;
innateNoiseLvl = 0.01;
testNoiseLvl = 0.01;
initInputWind = 150;
initInputStart = 200;
sigStimEnd = initInputStart+initInputWind;
HighResSampleRate = 100;
scalingFactor = .66;
scalingTics = .2;
originalTonicLvl = ExExTrainTonicStims(1);
targLen = 800; %peak time of original target in # of time steps (total time is targLen + 200)
saveFileNameSuff = '';
dateString = datestr(clock, 30);
originalSaveFolder = '~/Documents/Data/TemporalInvariance';
subExpSaveFolder = 'ParamTest/ScaleFactor/Tau_10';
expSaveFolder = fullfile(originalSaveFolder, subExpSaveFolder, dateString);
if ~exist(expSaveFolder, 'dir')
    mkdir(expSaveFolder);
end

save(fullfile(expSaveFolder, 'instance variables.mat'));
for seedIndex = 1:length(seeds)
    
    tempRunSeed = seeds(seedIndex);
    seedStr = strcat(int2str(tempRunSeed),'_seed_');
    ratSaveFolder = fullfile(expSaveFolder, seedStr);
    if ~exist(ratSaveFolder, 'dir')
        mkdir(ratSaveFolder);
    end
    
    saveName = fullfile(ratSaveFolder,...
        strcat(int2str(testExNum),'PlstU_'));
    %     tonicStims = trainTonicStims;
    %     numStim = length(ExExTrainTonicStims);
    RRN_Plastic_RUNGPU;
end
T3 = toc;
