%USED TO RUN RRN_Plastic_GOD
%Deterime in Recurrene plasticity is more stable.
%RRN_Plastic_RUNExperC = no steady state input In(1) = 0

numExp = 1;

% SEED = [110 123433 45 98 123 1245 9880 38572 91866 98997 432 4112 5112 551225 12662 788377];
tempseed = 0; % used to make sure different seed when reading READ_W

% figure(5)
% set(gcf,'position',[400   100   672   504])
for expNum = 1:numExp
    SEED(expNum) = 99*expNum;
    %    SEED = 282051;
    %SEED = 289872;
    %SEED = 390258;
    %    SEED = 110;
    fprintf('         EXPERIMENT = %d\n',expNum);
    %    seed = SEED(expNum);
    seed =  99*tempRunSeed; %for NFH script runs


    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% RECURRENT PLASTICITY
    %get target Ex "INNATE"
    TRAIN_SWITCH = 0;
    TRAIN_RECURRENT_SWITCH = 0;
    READW_SWITCH = 0;
    TEACHER_FORCING = 0;
    %    TrainLoops = 1;
    TrainLoops = length(initStims);
    NOFEEDBACK = 1;
    NoiseAmp = innateNoiseLvl;
    PURE_TEST = 0;
    RRN_Plastic;
    EXTARGET = historyEx;
    OUTTARGET = historyOut;
    save(strcat(ratSaveFolder, 'ExTarget'), 'EXTARGET', 'OUTTARGET', 'trainStart', 'historyIN')
    save(strcat(ratSaveFolder, 'WExEx'), 'WExEx')

    %train WExEx Recurrent Weights
    'RRN training'
    TRAIN_SWITCH = 0;
    TRAIN_RECURRENT_SWITCH = 1;
    READW_SWITCH = 0;
    NOFEEDBACK = 1;
    NoiseAmp = recNoiseLvl;%0.001;
    TEACHER_FORCING = 0;
    PURE_TEST = 0;
    RRN_Plastic;
%     save(strcat(saveName, 'Rec train test variables'),'WOutEx', 'WExEx', 'WExExInit', 'WExOut', 'WInEx', 'historyEX', 'EXTARGET', 'trainStart', 'historyIN');
%     for i = 1:length(Figures)
%         saveas(Figures{i}, strcat(saveName, '_RecTrainFig_Stim', int2str(i)), 'fig');
%     end

    %       %train WExOut %normal target
    'WExOut'
    TRAIN_SWITCH = 1;
    TRAIN_RECURRENT_SWITCH = 0;
    READW_SWITCH = 1;
    NOFEEDBACK = 1;
    NoiseAmp = WExOutNoiseLvl;%0.001;
    PURE_TEST = 0;
    RRN_Plastic;
%     save(strcat(saveName, 'ExOut train test variables'),'originalTarget','WOutEx', 'WExEx', 'WExExInit', 'historyOUT','WExOut', 'WInEx', 'historyEX', 'trainStart','historyIN');
%     for i = 1:length(Figures)
%         saveas(Figures{i}, strcat(saveName, '_ExOutTrainFig_Stim', int2str(i)), 'fig');
%     end

    %Test %full target
    TRAIN_SWITCH = 0;
    TRAIN_RECURRENT_SWITCH = 0;
    READW_SWITCH = 1;
    NOFEEDBACK = 1;
    NoiseAmp = testNoiseLvl;
    TEACHER_FORCING = 0;
    PURE_TEST = 0;
    TrainLoops = length(testTonicStims)*10;
    RRN_Plastic;
    OUT(expNum).RP = historyOUT;
    save(strcat(saveName, ' test output variables'), 'WExEx', 'WExExInit','WExOut', 'WInEx', 'historyEX', 'historyOUT', 'historyExOld', 'testOutTargets', 'historyIN');
    for i = 1:length(Figures)
        saveas(Figures{i}, strcat(saveName, '_TestOutputFig_Stim', int2str(i)), 'fig');
    end

end
