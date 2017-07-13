%{
Class for training and testing RNNs
%}


classdef RNN < handle
    
    properties (Access = protected)

        % state and network variables
        % WExEx = recurrent connectivity matrix, WExOut = output connectivity
        % matrix, WInEx = Input connectivity matrix, Ex = unit output
        % (transformed), ExV = unit input (not transformed), Out = output
        % state, In = input state
        %         
        % g_ = GPU variable

        WExEx; WExOut; WInEx; Ex; ExV; Out; In; WExExInit; % initialize weight matrices
        g; % network gain
        gEx; gExV; gnoiseArrPlusIn; % initialize activity variables
        gWExEx; gWExOut; % GPU weight matrices
        NetworkSeed; % current network seed for random number generator
        InitSeed; % seed used to initiate network
        LastSeed; % previous seed used for rng
        Date; % date RNN was created
        Name; % RNN identifier
        %Training variables
        % CPU variables
        PRec; PreSyn;  % p-matrices for RLS learning rule
        dW; P;
        Target; % target for recurrent training
        Trained_RNN; Trained_Output; % trained or untrained boolean
        % GPU training variables
        PRecs; % P matrix
        PIndsC; % presynaptic input sources for each neuron
        cumPreSizesC; % store
        cumPreSizeSq; % size of p-matrix for each unit
        mxPre; % largest # of synapses onto single unit
        cumPreSizes; % array of cumulative # of pre. syn. units. for reconstructing WExEx
        PInds; % indices of all pre. syn. units
        ExList; % list of RNN unit numbers for GPU
        RNNTrainTrials=0;
    end
    
    properties
        % simulation parameters, initialized to default values
        TrainStimOrder;
        alphaParam = 10; %10;
        cellsPerGridCol = 4;
        numIn = 3;
        P_Connect = 0.2;
        tau = 50;
        numOut = 1;
        numEx; %total number of Ex units
        TrigAmp; %stim level for initializeing RRN activity. uses WIn(stim)
        ExExTrainTonicStims;%stims level 2 for tonic RRN stim uses WIn(3)
        ExOutTrainTonicStim;
        innateNoiseLvl = 0.05;
        TrigDur;
        scalingFactor;
        scalingTics;
        originalTonicLvl;
        TargLen; %peak time of original target in # of time steps (total time is targLen + 200)
        HighResSampleRate = 100;
        scaleDir = 1;
        restDur=30000;
        trainTimeStep=1;
        CueIn=2;
        SpeedIn=3;
    end
    
    methods
        function obj = RNN(InitSeed, G, numEx, numIn, numOut)
            if nargin > 0
                obj.InitSeed = InitSeed;
                obj.g = G; obj.numEx = numEx; obj.numIn = numIn;
                obj.numOut = numOut;
                seed = obj.InitSeed * 99;
                obj.newState(seed)
                obj.initStateRRN
                obj.instantiateRNN
                obj.Date = datestr(clock, 30);
            end
        end
        
        function [newExV, newEx] = IterateRNN_CPU(obj,In)
            if  isempty(obj.Ex) || isempty(obj.ExV)
                error('RNN state must be set')
            else
                ex_input = obj.WExEx'*obj.Ex + In;
                newExV = obj.ExV + (-obj.ExV + ex_input)./obj.tau;
                newEx = tanh(newExV);
                obj.Ex = newEx;
                obj.ExV = newExV;
            end
        end
        
        function [Out] = IterateOutCPU(obj)
            Out = obj.WExOut'*obj.Ex;
            obj.Out = Out;
        end
        
        function [newExV, newEx] = IterateRNN_GPU(obj, NoisePlusIn, t)
            gInputPlusNoise = gpuArray(single(NoisePlusIn));
            tgWExEx = obj.gWExEx;
            tgWExOut = obj.gWExOut;
            tPInds = obj.PInds;
            tEx = obj.gEx;
            tExV = obj.gExV;
            tOut = obj.Out;
            
            obj.iterateRNN_GPUFull(tgWExEx, ...
                tgWExOut, ...
                gInputPlusNoise, ...
                obj.cumPreSizes, ...
                tPInds, ...
                tEx, ...
                tExV, ...
                tOut, ...
                single(t), ...
                single(obj.numEx/obj.cellsPerGridCol), ...
                single(obj.cellsPerGridCol), ...
                single(obj.numOut), ...
                single(obj.tau), ...
                single(obj.mxPre))
            
            obj.gEx = tEx; obj.gExV = tExV; % repopulate RNN's state after end of training
            obj.Out = tOut; % continued
            newExV = gather(obj.gExV); newEx = gather(obj.gEx); % function output
        end
        
        function [hRNN] = trainFORCE(obj, Target, TrainWindow,...
                InputPlusNoise)
            tmax = size(InputPlusNoise, 2);
            hRNN = zeros(obj.numEx, tmax);
            
            for t = 1:tmax
                [~, hRNN(:, t)] = obj.IterateRNN_CPU(InputPlusNoise(:,t));
                
                if ismember(t, TrainWindow)
                    error_rec = obj.Ex - Target(:, t);
                    WExExNew = obj.WExEx;
                    for i=obj.numEx %loop through Post Ex
                        %From Sussillo and Abbott Code
                        preind = obj.PreSyn(i).ind;
                        %ex = historyRNN(preind,t-1); % Rodrigo's training algorithm
                        ex = hRNN(preind,t);
                        k = obj.PRec(i).P*ex;
                        expex = ex'*k;
                        c = 1.0/(1.0 + expex);
                        obj.PRec(i).P = obj.PRec(i).P - k*(k'*c); % has to store this for later
                        dw = error_rec(i)*k*c;
                        WExExNew(preind,i) = obj.WExEx(preind,i) - dw;
                    end
                    obj.WExEx = WExExNew;
                end
            end
            obj.RNNTrainTrials=obj.RNNTrainTrials+1;
        end
        
        function trainRNNTargetGPU(obj, Target, TrainWindow,...
                InputPlusNoise)
            TMax = size(InputPlusNoise, 2);
            hRNN = gpuArray(single(zeros(obj.numEx, TMax)));
            gInputPlusNoise = gpuArray(single(InputPlusNoise));
            tEx = obj.gEx; tExV = obj.gExV; tWExEx = obj.gWExEx; % gather RNN's state variables to start training
            tOut = gpuArray(obj.Out); tWExOut = obj.gWExOut; tPInds = obj.PInds; % same
            tPRecs = obj.PRecs; tcumPreSizes = obj.cumPreSizes;% same
            tcumPreSizesSq = obj.cumPreSizeSq;
            for t  = 1:TMax % time loop for training
                obj.iterateRNN_GPUFull(tWExEx, ...
                    tWExOut, ...
                    gInputPlusNoise, ...
                    tcumPreSizes, ...
                    tPInds, ...
                    tEx, ...
                    tExV, ...
                    tOut, ...
                    single(t), ...
                    single(obj.numEx/obj.cellsPerGridCol), ...
                    single(obj.cellsPerGridCol), ...
                    single(obj.numOut), ...
                    single(obj.tau), ...
                    single(obj.mxPre)); % iterate the RNN state on the GPU
                
                %hRNN(:, t) = tEx;
                if ismember(t, TrainWindow) % check if in training window
                    error_rec = tEx - Target(:,t); % calculate error from target
                    obj.trainRNNFORCE_GPU(tEx,...
                        error_rec,...
                        tcumPreSizes,...
                        tcumPreSizesSq,...
                        tPInds,...
                        tPRecs,...
                        tWExEx,...
                        obj.ExList,...
                        obj.numEx/obj.cellsPerGridCol,...
                        obj.cellsPerGridCol,...
                        obj.mxPre) % update weights from recorded error
                end
            end
            obj.gEx = tEx; obj.gExV = tExV; obj.gWExEx = tWExEx; % repopulate RNN's state after end of training
            obj.Out = tOut; obj.gWExOut = tWExOut; obj.PInds = tPInds; % continued
            obj.PRecs = tPRecs; % continued
            obj.RNNTrainTrials=obj.RNNTrainTrials+1;
        end
        
        function trainOutputFORCE(obj, outTarget)
            error_minus = obj.Out - outTarget;
            %From Sussillo and Abbott Code
            k = obj.P*obj.Ex;
            ExPEx = obj.Ex'*k;
            c = 1.0/(1.0 + ExPEx);
            obj.P = obj.P - k*(k'*c);
            dw = error_minus*k*c;
            obj.WExOut = obj.WExOut - dw;
            %obj.dW = sum(abs(dw));
        end
        
        function [scaledTargs, times] = scaleTarget(obj, target,...
                trainStart, activeRRNEnd)
            SpeedSignals = obj.ExExTrainTonicStims;
            numScaledTargs = length(SpeedSignals);
            times = zeros(1,numScaledTargs);
            trainingExTarget = target(:, trainStart+1:activeRRNEnd);
            preTrainTarget = target(:,1:trainStart);
            postActiveTraget = target(:,activeRRNEnd+1:end);
            highResTarget = interp1(trainingExTarget',...
                [1/obj.HighResSampleRate:1/obj.HighResSampleRate:activeRRNEnd-trainStart]);
            if size(highResTarget,1) > 1
                highResTarget = highResTarget';
            end
            
            for inAmpInd = 1:numScaledTargs
                currentStim = SpeedSignals(inAmpInd);
                % changed 2/8/17 to force scaling proportional to factor of
                % orginalTonicLvl
                sampleFactor=(currentStim/obj.originalTonicLvl)^sign(obj.scaleDir);
                newExTargSample = round([1:...
                    sampleFactor:...
                    activeRRNEnd-trainStart]*obj.HighResSampleRate);
                sampledTarget = highResTarget(:, newExTargSample);
                newTarget = [preTrainTarget sampledTarget postActiveTraget];
                scaledTargs{inAmpInd} = newTarget;
                times(inAmpInd) = size(newTarget,2);
            end
        end
        
        function [gatedTarget] = gatedExTarget(obj, EXTARGET, restTime, tau)
            totalTargLen = size(EXTARGET,2);
            if (totalTargLen - restTime) < tau^2
                error('EXTARGET must the tau^2 longer than restTime to ensure 0 state', class(RNN))
            else
                ExTargetMask = ones(1,totalTargLen);
                ExTargetMask(restTime:end)= exp(-([restTime:totalTargLen]-restTime)/(2*tau));
                ExTargetMask(ExTargetMask>1)=1;
                gatedTarget = EXTARGET.*repmat(ExTargetMask,obj.numEx,1);
            end
        end
        
        function [Input] =...
                generateInputPulses(obj, Units, Amps, Starts, Ends, TMax)
            Input = zeros(obj.numIn, TMax);
            for i = 1:numel(Units)
                thisUnit = Units(i);
                thisAmp = Amps(i);
                thisRange = Starts(i):Ends(i);
                Input(thisUnit, thisRange) = thisAmp;
            end
        end
        
        function [noiseArr] =...
                generateNoiseInput(obj, Input, noiseAmp)
            %{
            Adds gaussian noise to an input matrix.
            noiseAmp = amplitude of noise (width of the gaussian)
            Input = matrix of input to the RRN during a trial, generated by
            the generateInputPulses function
            %}
            tmax = size(Input,2);
            noiseArr = noiseAmp*randn(obj.numEx,tmax) + obj.WInEx * Input;
        end
        
        function saveRNN(obj, Directory)
            NetNameStr = ['Seed', num2str(obj.InitSeed), 'RNN'];
            obj.Name=NetNameStr;
            saveName = fullfile(Directory, [NetNameStr,'_',obj.Date]);
            eval([NetNameStr ' = obj;']);
            
            if ~isempty(obj.Target)
                targetFile = [saveName, '_RNNTarget'];
                target = obj.Target;
                obj.Target = [];
                save(targetFile, 'target')
                clear target
            end
            save(saveName, NetNameStr, '-v7.3')
            clear(NetNameStr)
        end
        
        function saveWeights(obj,Directory)
            NetNameStr = ['Seed', num2str(obj.InitSeed), 'RNN'];
            obj.Name=NetNameStr;
            saveName = fullfile(Directory, [NetNameStr,'_',obj.Date]);
            eval([NetNameStr ' = obj;']);
            WFile = [saveName,'_Weights'];
            WExEx=obj.WExEx;WExExInit=obj.WExExInit;WInEx=obj.WInEx;
            WExOut=obj.WExOut;
            save(WFile,'WExEx','WExExInit','WInEx','WExOut')
        end
        
        function initStateRRN(obj)
            obj.Ex   = rand(obj.numEx,1)*2-1;
            obj.ExV  = zeros(obj.numEx,1);
            obj.Out  = zeros(obj.numOut,1);
            obj.In   = zeros(obj.numIn,1);
        end
        
        function zeroStateRRN(obj)
            obj.Ex   = zeros(obj.numEx,1);
            obj.ExV  = zeros(obj.numEx,1);
            obj.Out  = zeros(obj.numOut,1);
            obj.In   = zeros(obj.numIn,1);
        end
        
        function randStateRRN(obj)
            obj.Ex   = rand(obj.numEx,1);
            obj.ExV  = zeros(obj.numEx,1);
            obj.Out  = zeros(obj.numOut,1);
            obj.In   = zeros(obj.numIn,1);
        end
        
        function RNNStateGPU(obj)
            obj.gEx = gpuArray(single(obj.Ex));
            obj.gExV = gpuArray(single(obj.ExV));
        end
        
        function clearStateVars(obj)
            obj.gEx=[]; obj.gExV=[]; obj.gWExEx=[]; obj.gWExOut=[];
            obj.PRecs=[]; obj.cumPreSizeSq=[]; obj.cumPreSizes=[];
            obj.PInds=[]; obj.ExList=[]; obj.PIndsC=[]; obj.PInds=[];
            obj.PRec=[]; obj.PreSyn=[]; obj.cumPreSizesC=[]; obj.Ex=[];
            obj.ExV=[]; obj.P=[]; obj.Out=[]; obj.In=[];
        end
        
        function generate_W_P_GPU(obj)
            obj.PIndsC = [];
            obj.cumPreSizesC = [];
            obj.cumPreSizeSq = 0;
            obj.mxPre = 0;
            Ws = [];
            %{
			Sort through recurrent weights and get all synapse locations.
			Store those locations and their weights in a single array.
			This is will be passed to the GPU
            %}
            for i=1:obj.numEx
                ind = find(obj.WExEx(:,i))'; % find presynaptic indices
                obj.PIndsC = [obj.PIndsC ind]; % store all presyn. inds. in single array
                Ws = [Ws obj.WExEx(ind,i)']; % store the weights for those synapses
                obj.cumPreSizesC = [obj.cumPreSizesC length(obj.PIndsC)]; % store # of synapses onto each unit
                if length(ind) > obj.mxPre % store largest # of synapses onto single unit
                    obj.mxPre = length(ind);
                end
                obj.cumPreSizeSq = obj.cumPreSizeSq + length(ind)^2; % squared # of synapses. for P-matrix
            end
            obj.PRecs = zeros(1, obj.cumPreSizeSq); % instantiate p-matrix
            obj.cumPreSizeSq=[]; % reset squared synapse size
            stP = 1; % # of items in P matrix
            
            for i=1:obj.numEx % loop through every unit
                if i > 1 % if after first unit, get number of presynaptic units by taking difference cumPreSizesC between current unit and previous
                    en = obj.cumPreSizesC(i) - obj.cumPreSizesC(i-1);
                else % if first unit, take first item
                    en = obj.cumPreSizesC(i);
                end
                PTmp = eye(en)/obj.alphaParam; % for each unit's synapses make p-matrix initial state with alpha (forgetting rate of RLS learning rule)
                obj.PRecs(stP:(stP+en^2-1)) = PTmp(:)'; % set p-matrix values to initial forgetting rate
                obj.cumPreSizeSq = [obj.cumPreSizeSq stP+en^2-1]; % store size of p-matrix for each unit within PRecs
                stP = stP+en^2; % iterate squared index
                %[i stP-1 length(obj.PRecs)] % display progress if desired, mostly for debugging
            end
            % convert variables to GPU arrays of type single
            obj.PRecs = gpuArray(single(obj.PRecs));
            obj.PInds = gpuArray(int32(obj.PIndsC));
            obj.cumPreSizes = gpuArray(int32(obj.cumPreSizesC));
            obj.cumPreSizeSq = gpuArray(int32(obj.cumPreSizeSq));
            obj.P = gpuArray(single(eye(obj.numEx)/obj.alphaParam));
            obj.gWExEx = gpuArray(single(Ws));
            obj.gWExOut = gpuArray(single(obj.WExOut));
            
            obj.ExList = gpuArray(int32(sort(1:floor(obj.numEx))));
            obj.gEx = gpuArray(single(obj.Ex));
            obj.gExV = gpuArray(single(obj.ExV));
        end
        
        function generateP_CPU(obj)
            for i=1:obj.numEx
                obj.PreSyn(i).ind = find(obj.WExEx(:,i));
                obj.PRec(i).P = eye(length(obj.PreSyn(i).ind))/obj.alphaParam;
            end
            obj.P   = eye(obj.numEx)/obj.alphaParam;
        end
        
        function reconWs(obj)
            Ws = gather(obj.gWExEx);
            i1 = 0;
            for i=1:obj.numEx
                i2 = obj.cumPreSizesC(i);
                len = i2-i1;
                obj.WExEx(obj.PIndsC(i1+(1:len)),i) = Ws(i1+(1:len))';
                i1 = i2;
            end
            obj.WExOut=gather(obj.gWExOut);
        end
        
        function newState(obj, seed)
            obj.LastSeed = obj.NetworkSeed;
            obj.NetworkSeed = seed;
            rng(obj.NetworkSeed);
        end
        
        function Seed = getNetworkSeed(obj)
            Seed=obj.InitSeed;
        end
        
        function newWExOut(obj)
            obj.makeWExOut;
        end
        
        function [DateStr] = getDate(obj)
            DateStr=obj.Date;
        end
        
        function [Name]=getName(obj)
            Name = ['Seed', num2str(obj.InitSeed), 'RNN']; % add this to instatiation block
            obj.Name=Name;
        end
        
        function setRNNTarget(obj, Target)
            obj.Target = Target;
        end
        
        function setWExEx(obj, WExExNew)
            weightDim=size(WExExNew); % check if WExEx is correct dimensions
            if ~(numel(weightDim)==2) ||...
                    ~isempty(find(weightDim~=obj.numEx))
                error('WExExNew is the wrong size! Weights not changed')
            else
                obj.WExEx = WExExNew;
                obj.Date = datestr(clock, 30);
            end
        end
        
        function [target] = getRNNTarget(obj)
            target = obj.Target;
        end
        
        function [RecW] = getWExEx(obj)
            RecW = obj.WExEx;
        end
        
        function [WExOut] = getWExOut(obj)
            WExOut = obj.WExOut;
        end
        
        function [RecWInit] = getWExExInit(obj)
            RecWInit = obj.WExExInit;
        end
        
        function [WIn] = getWInEx(obj)
            WIn = obj.WInEx;
        end
        
        function setWInEx(obj, WIn)
            if isequal([obj.numEx,obj.numIn], size(WIn))
                obj.WInEx = WIn;
            else
                error('Error. \nWIn must be size NxIn', class(RNN))
            end
        end
        
        function [newEx]=setExV(obj,newExV)
            %{
            Sets RNN ExV to new ExV, and Ex to tanh(ExV)
            %}
            if isequal(size(newExV),[obj.numEx,1])
                obj.ExV=newExV;
                newEx=tanh(obj.ExV);
                obj.Ex=newEx;
                fprintf([obj.Name,' ExV set, Ex set to tanh(ExV)\n'])
            else
                error('newEx must be size [numEx,1]. Ex not set.')
            end
        end
        
%         function obj=loadobj(s)
%             if isstruct(s)
%                 neObj=RNN
%             end
%         end
    end
    
    methods (Access = protected)
        function instantiateRNN(obj)
            obj.makeWExEx;
            obj.makeWExOut;
            obj.makeWInEx;
            obj.WExExInit = obj.WExEx;
        end
        
        function makeWExEx(obj)
            WMask = rand(obj.numEx, obj.numEx);
            WMask(WMask<(1-obj.P_Connect))=0;
            WMask(WMask>0) = 1;
            WExExtmp = randn(obj.numEx, obj.numEx)*sqrt(1/(obj.numEx*obj.P_Connect));
            WExExtmp = obj.g*WExExtmp.*WMask;
            WExExtmp(1:(obj.numEx+1):obj.numEx*obj.numEx)=0;
            obj.WExEx = WExExtmp;
        end
        
        function makeWExOut(obj)
            obj.WExOut = randn(obj.numEx, obj.numOut)*sqrt(1/obj.numEx);
        end
        
        function makeWInEx(obj)
            obj.WInEx = randn(obj.numEx, obj.numIn);
        end
        
        function iterateRNN_GPUFull(obj, gWExEx, gWExOut, NoisePlusIn,...
                cumPreSizes, PInds, Ex, ExV, Out, t, NumCol,...
                cellsPerGridCol, numOut, tau, mxPre)
            
            mexGPUUpdate(gWExEx,...
                gWExOut,...
                NoisePlusIn,...
                cumPreSizes,...
                PInds,...
                Ex,...
                ExV,...
                Out,...
                single(t),...
                single(NumCol),...
                single(cellsPerGridCol),...
                single(numOut), ...
                single(tau),...
                single(mxPre));
        end
        
        function trainRNNFORCE_GPU(obj, gEx, error_rec, cumPreSizes,...
                cumPreSizeSq, PInds, PRecs, gWExEx, ExList, NumCol,...
                cellsPerGridCol, mxPre)
            mexGPUTrain(gEx, ...
                error_rec, ...
                cumPreSizes, ...
                cumPreSizeSq, ...
                PInds, ...
                PRecs, ...
                gWExEx, ...
                ExList, ...
                single(NumCol), ...
                single(cellsPerGridCol), ...
                single(mxPre), ...
                single(length(ExList)));
        end
        
    end
end
