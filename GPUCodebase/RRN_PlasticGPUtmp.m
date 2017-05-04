%Sussilo and Abbott
%Firing rate model with tau = 10;
%you can train w/ TRAIN_SWITCH = 1 and READW_SWITCH = 0;
%and test with 0 and 1, respectively.
%WExEx(pre,post);

%%% Clear old variables
clear historyEx historyExOld historyOut historyOUT WExEx WInEx WExOut WOutEx Figures historyEX
clear P PRec PreSyn
clear EXTARGET currentExTarget
close all

%{
Classes of simulations:
Get Target and instantiate network:
    Generate Weights and perform one trial of activity, save weights and
    innate target(s)

Train RNN:
    Load weights, perform training

%}

%%% Set seed
rand('seed',seed);
randn('seed',seed);

%%%% Set simulation parameters
if TRAIN_SWITCH
    SimLoops = wExOutTrainTrials;
elseif (TRAIN_RECURRENT_SWITCH == 1)
    SimLoops = rrnTrainTrials;
    load(strcat(ratSaveFolder, 'ExTarget'));    %prestored target for all Ex units
end

RNNTarget = EXTARGET;
g = testG;
alphaParam = 10; %10;
trainStart = initInputStart+initInputWind;
cellsPerGridCol = 4;
numIn = 3;
P_Connect = 0.2;
tau   = 10;
numOut= 1;

%% Instantiate Weight matrices % initial conditions
[WExEx, WExOut, WInEx, Ex, ExV, Out] = instantiateRNN(P_Connect, g, numEx, numOut, numIn);
WExExInit = WExEx;

if TrainRNNSWITCH
    Ex   = gpuArray(single(Ex));
    ExV  = gpuArray(single(ExV));
    Out  = gpuArray(single(Out));
end
if READW_SWITCH
    load(strcat(ratSaveFolder, 'W_RRN_Plastic'));
    tempseed = seed+tempseed;
    rand('seed',tempseed);
    randn('seed',tempseed);
end

[PRecs, PIndsC, cumPreSizesC, cumPreSizeSq, mxPre, gWExEx, gWExOut, cumPreSizes, PInds, P] = ...
    generate_W_P_GPU(WExEx, numEx, alphaParam, WExOut);


%% INPUT AMPLITUDES: Second Value is Real Input
%%% InAmp2 is set based on the current training/testing paradigm
InAmp    = initStims; %changed 5/22 for constant input
if TRAIN_SWITCH == 0 && TRAIN_RECURRENT_SWITCH == 0 && READW_SWITCH == 0 %get innate target
    InAmp2 = originalTonicLvl;
elseif TRAIN_SWITCH == 0 && TRAIN_RECURRENT_SWITCH == 1 && READW_SWITCH == 0 % RRN training
    InAmp2 = ExExTrainTonicStims;
elseif TRAIN_SWITCH == 0 && TRAIN_RECURRENT_SWITCH == 0 && READW_SWITCH == 1 % testing
    InAmp2 = testTonicStims;
elseif TRAIN_SWITCH == 1 && TRAIN_RECURRENT_SWITCH == 0 && READW_SWITCH == 1 % train WExOut
    InAmp2 = ExOutTrainTonicStim;
end
numStim = length(InAmp2);
historyOUT = num2cell(1:numStim)';
historyEX = num2cell(1:numStim)';
historyIN = num2cell(1:numStim)';

%% generate the output target and scale output and RRN targets
activityEnd = targLen+200+trainStart; % end of the period of RRN trajectory

originalTarget = normpdf(1:activityEnd+trainStart,targLen+trainStart,25);
OutTargs = {(1/max(originalTarget))*originalTarget}; %normalize output target

if TRAIN_RECURRENT_SWITCH == 1
    if restEx
        RNNTarget = gatedExTarget(RNNTarget, restTime, tau, numEx);
    end
    [RNNTargs, TargLengths] = scaleTarget(RNNTarget, trainStart, activityEnd, InAmp2, scaleDir, scalingTics, originalTonicLvl, HighResSampleRate, scalingFactor);
    save(strcat(ratSaveFolder, 'RNNTargets'), 'RNNTargs', 'RNNTarget');
elseif TRAIN_SWITCH == 1
    [OutTargs, TargLengths] = scaleTarget(originalTarget, trainStart, activityEnd, InAmp2, scaleDir, scalingTics, originalTonicLvl, HighResSampleRate, scalingFactor);
    save(strcat(ratSaveFolder, 'OutTargs'), 'scaledOutTargs');
elseif TRAIN_SWITCH == 0 && TRAIN_RECURRENT_SWITCH == 0 && READW_SWITCH == 1
    [OutTargs, TargLengths] = scaleTarget(originalTarget, trainStart, activityEnd, InAmp2, scaleDir, scalingTics, originalTonicLvl, HighResSampleRate, scalingFactor);
end
%% Training
%%%% Find all target lengths, then set tmax to that plus some buffer
for loop = 1:SimLoops
    
    stim = mod(loop-1,numStim)+1;
    %%% SET tmax, trainTime, and target %%%
    if TRAIN_SWITCH == 0 && TRAIN_RECURRENT_SWITCH == 0 && READW_SWITCH == 0 %Innate
        target = OutTargs{stim};
        trainTime = TargLengths(stim);
        tmax = TargLengths(stim) + 1000;
    elseif TRAIN_SWITCH == 0 && TRAIN_RECURRENT_SWITCH == 0 && READW_SWITCH == 1 %Testing
        target = OutTargs{stim};
        trainTime = 1;
        tmax = TargLengths(stim) + 1000;
    elseif TRAIN_SWITCH == 0 && TRAIN_RECURRENT_SWITCH == 1 && READW_SWITCH == 0 %Train RRN
        trainTime = TargLengths(stim);
        tmax = TargLengths(stim) + 1000;
        currentExTarget = RNNTargs{stim};
        gcurrentEXTARGET = gpuArray(single(currentExTarget));
    else                                                             %Train WExOut
        target = OutTargs{stim};
        trainTime = TargLengths(stim);
        tmax = TargLengths(stim) + 1000;
    end
    TRAIN_WINDOW = [trainStart+1 trainTime]
    
    %% Define input pulses
    In   = zeros(numIn,tmax);
    tonicInputDur = tmax;
    if restEx
        tonicInputDur = tmax - restTime;
    end
    In(2,initInputStart:trainStart) = InAmp(1);
    In(3,initInputStart:tonicInputDur) = InAmp2(stim);
    noiseArr = NoiseAmp*randn(numEx,tmax) + WInEx*In;
    gnoiseArrPlusIn = gpuArray(single(noiseArr));
    ExList = gpuArray(int32(sort(1:floor(numEx)))); % + (stim-1)*25;
    
    %%% Preallocate
    historyEx=zeros(numEx,tmax);
    historyOut=zeros(numOut,tmax);
    historyIn =zeros(numIn,tmax);
    error_minus = zeros(1,tmax);
    error_plus = zeros(1,tmax);
    dW   = zeros(numEx,1);
    ghistoryEx=gpuArray(single(historyEx));
    ghistoryOut=gpuArray(single(historyOut));
    ghistoryIn =gpuArray(single(historyIn));
    
    %% Instatiate the netowrk state and begin the time loop
    if (TRAIN_SWITCH==0)
        fprintf('OUTPUT LEARNING IS OFF | LOOP = %3d(%d)\n',loop,stim);
    else
        fprintf('LOOP = %3d(%d)\n',loop,stim);
    end
    if (TRAIN_RECURRENT_SWITCH==1)
        fprintf('RECURRENT LEARNING IS ON | LOOP = %3d(%d)\n',loop,stim);
    end
    
    ExV = ExV*0;
    Ex  = Ex*0;
    Out = Out*0;
    
    %% random initial state
    ExV = gpuArray(single(2*rand(numEx,1)-1));
    
    
    %% Time Loop
    for t=1:(tmax + postTime)
        if rem(t,1000)==0, fprintf('t=%5d\n',t), end;
        %% set the input amplitudes for this time step depending on the
        %% input window
        %%% DYNAMICS HAPPENS HERE
        mexGPUUpdate(gWExEx,gWExOut, gnoiseArrPlusIn, cumPreSizes, PInds, Ex, ExV, Out, single(t), single(numEx/cellsPerGridCol), single(cellsPerGridCol), single(numOut), single(tau), single(mxPre));
        
        %%    TRAIN RECURRENT UNITS
        if (TRAIN_RECURRENT_SWITCH)
            if t>TRAIN_WINDOW(1) & t<TRAIN_WINDOW(2) % end of training
                error_rec = Ex - gcurrentEXTARGET(:,t);
                %            [PRecs gWExEx] = mexGPUTrain(Ex, error_rec, cumPreSizes, cumPreSizeSq, PInds, PRecs, gWExEx, numEx, mxPre);
                mexGPUTrain(Ex, error_rec, cumPreSizes, cumPreSizeSq, PInds, PRecs, gWExEx, ExList, single(numEx/cellsPerGridCol), single(cellsPerGridCol), single(mxPre), single(length(ExList)));
                
            end
        end
        
        
        
        
    end
    
    %RECONSTRUCT WExEx from Ws
    Ws = gather(gWExEx);
    i1 = 0;
    for i=1:numEx
        i2 = cumPreSizesC(i);
        len = i2-i1;
        WExEx(PIndsC(i1+(1:len)),i) = Ws(i1+(1:len))';
        i1 = i2;
    end
    WExOut=gather(gWExOut);
end

if (TRAIN_SWITCH == 1 || TRAIN_RECURRENT_SWITCH == 1)
    save(strcat(ratSaveFolder, 'W_RRN_Plastic'), 'WExOut', 'WExEx', 'WInEx', 'seed', 'g', 'alphaParam', 'P_Connect') %NoiseAmp
end
