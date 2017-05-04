clear;
tic
seeds = [110];
restEx = 1; % true/false for adding a "rest" period for training a gated attractor
restTime = 200; %time to train netowrk to rest for after innate trajectoy training

    % testExNums = [0,20,50,80,100,200,300,400,500,600];
    numEx = 2400; %total number of Ex units
    testExNum = 2400; %number of plastic Ex units at a time
    % seeds = [110 123433 45 98 123]; %seed to use to generate weights
    initStims = [2]; %stim level for initializeing RRN activity. uses WIn(stim)
    ExExTrainTonicStims = [0.35 0.15 0.55];%stims level 2 for tonic RRN stim uses WIn(3)
    ExOutTrainTonicStim = [0.35];
    testTonicStims = [0.35 0.15 0.55 0.25 0.45 0.05];
    rrnTrainTrials = 25; %number of iterations for training recurrent network
    wExOutTrainTrials = length(ExOutTrainTonicStim)*10; %number of iterations for training WExOut
    alternatingPlastExUnits = 0; %boolean for setting whether the plastic Ex units change with diferent stims
    shortInnateTarget = 1; %boolean for downsampling the innate target
    scaleDir = -1; %+/- 1 indicating whether an increase in tonic stim level scales the target up or down
    postTime = 150;
    randomInit = 1;
    testG = 1.6;
    recOnVar = 1;
    recNoiseLvl = 0.01;
    WExOutNoiseLvl = 0.01;
    innateNoiseLvl = 0.01;
    testNoiseLvl = 0.01;
    initInputWind = 50;
    initInputStart = 200;
    sigStimEnd = initInputStart+initInputWind;
    HighResSampleRate = 100;
    scalingFactor = .33333;
    scalingTics = .2;
    originalTonicLvl = .35;
    targLen = 750; %peak time of original target in # of time steps (total time is targLen + 200)
    saveFileNameSuff = '';
    dateString = datestr(clock, 30);
    originalSaveFolder = '/home/nhardy/Documents/Data/';
    subExpSaveFolder = 'GPU_Test/';
    expSaveFolder = strcat(originalSaveFolder, subExpSaveFolder, dateString, '/');
    if ~exist(expSaveFolder, 'dir')
        mkdir(expSaveFolder);
    end

    save(strcat(expSaveFolder, 'instance variables'), 'testExNum','seeds',...,
        'initStims','ExOutTrainTonicStim', 'ExExTrainTonicStims', 'testTonicStims',...
        'rrnTrainTrials','wExOutTrainTrials',...
        'alternatingPlastExUnits','shortInnateTarget','testG','recOnVar',...
        'recNoiseLvl','WExOutNoiseLvl','innateNoiseLvl','testNoiseLvl', 'numEx', 'scalingFactor',...
        'scalingTics', 'originalTonicLvl', 'targLen');
for seedIndex = 1:length(seeds)
    
        tempRunSeed = seeds(seedIndex);
        seedStr = strcat(int2str(tempRunSeed),'_seed_');
        ratSaveFolder = strcat(expSaveFolder, seedStr, '/');
        if ~exist(ratSaveFolder, 'dir')
            mkdir(ratSaveFolder);
        end

        saveName = strcat(ratSaveFolder,...
            int2str(testExNum),'PlstU_');
        %     tonicStims = trainTonicStims;
        %     numStim = length(ExExTrainTonicStims);
        RRN_Plastic_RUNGPU;
end
T3 = toc;
