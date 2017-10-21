classdef (ConstructOnLoad = false) RNN < handle
%RNN Class definition for RNN (Recurrent Neural Network)
%
% Contains methods for instantiating, training, and running the model The
% major focus is on innate learning (Laje and Buonomano, 2013), a learning
% rule based on reservoir computing theories of RNNs. This learning rule is
% unusual in that it trains the RNN to produce stable internal dynamics,
% indepentent of the targRec output. This has the advantage of making the
% activity of the RNN higher dimensional, allowing it to encode (and
% decode) more complex activity. In addition, this class implements
% temporal scaling of the recurrent dynamics, allowing the network to
% generalize its activity in time (Hardy et al., 2017).
%
% For more information see:
% 	Laje, R., and Buonomano, D.V. (2013). Robust timing and motor patterns
% 	by taming chaos in recurrent neural networks. Nat. Neurosci. 16,
% 	925�933.
%   Hardy, N.F., Goudar, V., Romero-Sosa, J.L., and Buonomano,
% 	D. (2017). A Model of Temporal Scaling Correctly Predicts that Weber?s
% 	Law is Speed-dependent. BioRxiv 159590.


    properties (Access = protected)
        % The user shouldn't need to change these properties after
        % instantiation
        % Network variables
        wRec;      % Recurrent weights
        wRecOut;     % Output weights
        WInRec;      % input Weights
        aRec;         % Recurrent unit output at time t (nonlinear transform of aRecS)
        aRecS;        % Recurrent unit state  at time t
        aOut;        % Output unit activity
        aIn;         % aIn unit activity
        wRecInit;  % store the initial weights
        g;          % network gain
        % some variables for the GPU kernal
        gRec;  % initialize activity variables
        gaRecS;
        gnoiseArrPlusIn;
        gwRec; % GPU weight matrices
        gWRecOut;
        % Instantiation variables
        NetworkSeed; % current network seed for random number generator
        Seed; % seed used to initiate network
        LastSeed; % previous seed used for rng
        Date; % date RNN was created
        Name; % RNN identifier
        % CPU Training variables
        PRec;
		PreSyn;  % p-matrices for RLS learning rule
        dW;
		P;
        Target; % targRec for recurrent training
        Trained_RNN; Trained_Output; % trained or untrained boolean
        % GPU training variables
        PRecs; % P matrix
        PIndsC; % presynaptic input sources for each neuron
        cumPreSizesC; % store
        cumPreSizeSq; % size of p-matrix for each unit
        mxPre; % largest # of synapses onto single unit
        cumPreSizes; % array of cumulative # of pre. syn. units. for reconstructing wRec
        PInds; % indices of all pre. syn. units
        RecList; % list of RNN unit numbers for GPU
        RNNTrainTrials=0;
    end

    properties
        % simulation parameters, initialized to default values
        alpha       = 10; % damping parameter fro weight matrix
        nGridCell   = 4; % number of cells to cluster in the GPU
        nIn         = 3; % number of inputs
        pConn       = 0.2; % recurrent connection probability
        tau         = 50; % recurrent unit time constant
        nOut        = 1; % number of outputs
        nRec; %total number of aRec units
    end

    methods
        function obj = RNN(Seed, G, nRec, nIn, nOut)
          % Instantiate an RNN.
          % Inputs:
          %   Seed -  initial seed for the rng
          %   G - gain for the recurrent weights
          %   nRec - number of recurrent units
          %   nIn - number of inputs
          %   nOut - number of outputs

            if nargin > 0
                obj.Seed    = Seed;
                obj.g       = G;
                obj.nRec    = nRec;
                obj.nIn     = nIn;
                obj.nOut    = nOut;
                seed        = obj.Seed * 99;
                obj.newState(seed)
                obj.randStateRRN
                obj.instantiateRNN
                obj.Date = datestr(clock, 30);
            end
        end

        function [newaRecS, newRec] = IterateRNN_CPU(obj,aIn)
          % Iterate the RNN one time step using the CPU. Requires the input at t
          % aIn - nRec x nIn vector of input amplitudes
            if  isempty(obj.aRec) || isempty(obj.aRecS)
                error('RNN state must be set')
            else
                ex_input = obj.wRec'*obj.aRec + aIn;
                newaRecS   = obj.aRecS + (-obj.aRecS + ex_input)./obj.tau;
                newRec    = tanh(newaRecS);
                obj.aRec   = newRec;
                obj.aRecS  = newaRecS;
            end
        end

        function [aOut] = IterateOutCPU(obj)
          % update the output based on the current RNN state
            aOut = obj.wRecOut'*obj.aRec;
            obj.aOut = aOut;
        end

        function [newaRecS, newRec] = IterateRNN_GPU(obj, NoisePlusIn, t)
            % Iterate the RNN using the GPU kernal, over the period t.
            % Handles all the GPU variable declarations, and sets the state of the RNN to the final result of the iteration.
            % Inputs:
            %   NoisePlusIn - nRec x tMax matrix of input amplitudes, plus the noise. Usually generated with generateNoiseInput
            gaInNoise = gpuArray(single(NoisePlusIn));
            tgwRec  = obj.gwRec;
            tgWRecOut = obj.gWRecOut;
            tPInds   = obj.PInds;
            taRec      = obj.gRec;
            taRecS     = obj.gaRecS;
            taOut     = obj.aOut;

            obj.iterateRNN_GPUFull(tgwRec, ...
                tgWRecOut, ...
                gaInNoise, ...
                obj.cumPreSizes, ...
                tPInds, ...
                taRec, ...
                taRecS, ...
                taOut, ...
                single(t), ...
                single(obj.nRec/obj.nGridCell), ...
                single(obj.nGridCell), ...
                single(obj.nOut), ...
                single(obj.tau), ...
                single(obj.mxPre))

            obj.gRec = taRec; obj.gaRecS = taRecS; % repopulate RNN's state after end of training
            obj.aOut = taOut; % continued
            newaRecS = gather(obj.gaRecS); newRec = gather(obj.gRec); % function output
        end

        function [aRec] = trainRNNFORCE_CPU(obj,targRec,wndwTrain,aInNoise)
          % Train the RNN using the innate learning rule (Laje and Buonomano 2013).
          % Innate learning trains the RNN to produce the targRec activity, robust to
          % perturbations.
          % Uses the FORCE algorithm (Sussillo and Abbott, 2009), a recurrsive-least-squares
          % based rule to update the weights. Basically this relies on the correlation of
          % the presynaptic activity to solve the credit assignment problem.
          %
          % Inputs:
          %   targRec - nRec x tMax matrix with the targRec for the recurrent activity
          %   wndwTrain - time points to update the RNN
          %   aInNoise - nIn x tMax matrix with the input and noise

            tMax = size(aInNoise,2);
            aRec = zeros(obj.nRec,tMax);
            for t = 1:tMax
                [~, aRec(:, t)] = obj.IterateRNN_CPU(aInNoise(:,t));
                if ismember(t, wndwTrain)
                    error_rec = obj.aRec - targRec(:, t); % error
                    wRecNew = obj.wRec;
                    for i = 1:obj.nRec %loop through recurrent units
                        % adapted from Sussillo and Abbott (2009)
                        preind = obj.PreSyn(i).ind; % get stored presynaptic indices
                        ex = aRec(preind,t); % recurrent activity
                        k = obj.PRec(i).P*ex; % Update running estimate of correlation with presynaptic units
                        expex = ex'*k;
                        c = 1.0/(1.0 + expex);
                        obj.PRec(i).P = obj.PRec(i).P - k*(k'*c); % has to store this for later
                        dw = error_rec(i)*k*c; % credit assignment according to the calculated correlations
                        wRecNew(preind,i) = obj.wRec(preind,i) - dw; % update the weights
                    end
                    obj.wRec = wRecNew; % update wRec
                end
            end
            obj.RNNTrainTrials=obj.RNNTrainTrials+1; % store the number of trained trials
        end

        function trainRNNFORCE_GPU(obj,Target,wndwTrain,aInNoise)
            % Similar to trainRNNFORCE_CPU, optimized for using the GPU

            tMax       = size(aInNoise, 2);
            aRec       = gpuArray(single(zeros(obj.nRec,tMax)));
            gaInNoise = gpuArray(single(aInNoise));
            taRec       = obj.gRec; % gather RNN's state variables to start training
            taRecS     = obj.gaRecS;
            twRec      = obj.gwRec;
            taOut       = gpuArray(obj.aOut);
            tWRecOut   = obj.gWRecOut;
            tPInds    = obj.PInds; % indices for the P matrix calculation
            tPRecs     = obj.PRecs;
            tcumPreSizes   = obj.cumPreSizes; % presynaptic weight sizes for the GPU kernal
            tcumPreSizesSq = obj.cumPreSizeSq;
            for t  = 1:tMax % time loop for training
                obj.iterateRNN_GPUFull(twRec, ...
                    tWRecOut, ...
                    gaInNoise, ...
                    tcumPreSizes, ...
                    tPInds, ...
                    taRec, ...
                    taRecS, ...
                    taOut, ...
                    single(t), ...
                    single(obj.nRec/obj.nGridCell), ...
                    single(obj.nGridCell), ...
                    single(obj.nOut), ...
                    single(obj.tau), ...
                    single(obj.mxPre)); % iterate the RNN state on the GPU

                if ismember(t, wndwTrain) % check if in training window
                    error_rec = taRec - Target(:,t); % calculate error from targRec
                    obj.trainRNNFORCE_GPUFull(taRec,...
                        error_rec,...
                        tcumPreSizes,...
                        tcumPreSizesSq,...
                        tPInds,...
                        tPRecs,...
                        twRec,...
                        obj.RecList,...
                        obj.nRec/obj.nGridCell,...
                        obj.nGridCell,...
                        obj.mxPre) % update weights from recorded error
                end
            end
            obj.gRec = taRec; % repopulate RNN's state after end of training
            obj.gaRecS = taRecS;
            obj.gwRec = twRec;
            obj.aOut = taOut;
            obj.gWRecOut = tWRecOut;
            obj.PInds = tPInds;
            obj.PRecs = tPRecs;
            obj.RNNTrainTrials = obj.RNNTrainTrials+1;
        end

        function trainOutFORCE(obj, outTarget)
            error_minus = obj.aOut - outTarget;
            % based on Sussillo and Abbott 2009
            k = obj.P*obj.aRec;
            RecPRec = obj.aRec'*k;
            c = 1.0/(1.0 + RecPRec);
            obj.P = obj.P - k*(k'*c);
            dw = error_minus*k*c;
            obj.wRecOut = obj.wRecOut - dw;
            %obj.dW = sum(abs(dw));
        end

        function [targNew, tTargs] = scaleTarget(obj, targRec, tStart, tEnd, scaleDir, rSample, aIn, aInBL)
            % Scale the innate target to new durrations based on the input amplitude.  Uses
            % the input amplitudes in aIn to determine the new durations, then scales the
            % passed targRec using linear interpolation.
            %
            % Inputs: targRec - nRec x tMax matrix containing the target recurrent
            % activity
            % tStart - start time of the target which will be scaled
            % tEnd - time of target
            % scaleDir - +/- 1; + = shorter durrations with increasing aIn, vice versa for -
            % rSample - number of points to upsample the target
            % aIn - input amplitudes for the new targets
            % aInBL - defines the baseline input amplitude to determine increase/decrease in duration

            nTargNew     = length(aIn);
            tTargs       = zeros(1,nTargNew);
            targRecTrain = targRec(:, tStart+1:tEnd); % store actual target
            preTarg      = targRec(:,1:tStart); % target before training
            postTarg     = targRec(:,tEnd+1:end); % post training
            highResTarget = interp1(targRecTrain',[1/rSample:1/rSample:tEnd-tStart]); % upsample the target
            if size(highResTarget,1) > 1
                highResTarget = highResTarget';
            end

            for iInAmp = 1:nTargNew
                tInA            = aIn(iInAmp);
                tScale          = (tInA/aInBL)^sign(scaleDir); % scaling factor
                ixSample        = round([1:tScale:tEnd-tStart]*rSample); % indices to sample highResTarget
                sampledTarget   = highResTarget(:, ixSample);
                targNew{iInAmp} = cat(2,preTarg,sampledTarget,postTarg);
                tTargs(iInAmp)  = size(targNew{iInAmp},2);
            end
        end

        function [gatedTarget] = gatedRecTarget(obj, targRec, tRest, tau)
            % Produces a target "gated attractor" training. Basically this decays the activity
            % in the target to 0 so that the RNN won't be spontaneously active (i.e. the
            % largest real eigenvalue of wRec is <= 1). Training with this type of
            % target prodces a network that responds only to a trained input
            %
            % Inputs:
            %     targRec - recurrent target
            %     tRest - duration of the 0-amplitude activity
            %     tau - time constant for the the exponential decay

            tTargDur = size(targRec,2);
            if (tTargDur - tRest) < tau^2
                error('targRec must the tau^2 longer than tRest to ensure 0 state', class(RNN))
            else
                targMask = ones(1,tTargDur);
                targMask(tRest:end)= exp(-([tRest:tTargDur]-tRest)/(2*tau));
                targMask(targMask>1)=1;
                gatedTarget = targRec.*repmat(targMask,obj.nRec,1);
            end
        end

        function [aInNew] = generateInputPulses(obj, ixIn, aIn, tStart, tEnd, tMax)
            % Helper function to generate input amplitudes for a defined duration`
            %
            % Inputs:
            %     ixIn - indices of the input units
            %     aIn - amplitudes for each unit
            %     tStart - start times for each unit
            %     tEnd - end times
            %     tMax - total time

            aInNew = zeros(obj.nIn, tMax);
            for i = 1:numel(ixIn)
                thisUnit = ixIn(i);
                thisAmp = aIn(i);
                thisRange = tStart(i):tEnd(i);
                aInNew(thisUnit, thisRange) = thisAmp;
            end
        end

        function [noiseArr] = generateNoiseInput(obj, aIn, noiseAmp)
            % Genreates the nRec x tMax weighted input matrix, plus Gaussian noise
            % Pass this to the GPU training functions
            % aInputs:
            %   noiseAmp = amplitude of noise (width of the gaussian)
            %   aIn = matrix of input to the RRN during a trial, generated by
            %   the generateaInputPulses function

            tMax = size(aIn,2);
            noiseArr = noiseAmp*randn(obj.nRec,tMax) + obj.WInRec * aIn;
        end

        function saveRNN(obj, dir)
            % Save the RNN object
            NetNameStr = ['Seed', num2str(obj.Seed), 'RNN'];
            obj.Name   = NetNameStr;
            saveName   = fullfile(dir, [NetNameStr,'_',obj.Date]);
            eval([NetNameStr ' = obj;']);

            if ~isempty(obj.Target)
                targetFile = [saveName, '_RNNTarget'];
                targRec = obj.Target;
                obj.Target = [];
                save(targetFile, 'targRec')
                clear targRec
            end
            save(saveName, NetNameStr, '-v7.3')
            clear(NetNameStr)
        end

        function saveWeights(obj,dir)
            % save the RNN weghts for easier access

            NetNameStr = ['Seed', num2str(obj.Seed), 'RNN'];
            obj.Name   = NetNameStr;
            saveName   = fullfile(dir, [NetNameStr,'_',obj.Date]);
            eval([NetNameStr ' = obj;']);
            WFile      = [saveName,'_Weights'];
            wRec       = obj.wRec;
            wRecInit   = obj.wRecInit;
            WInRecS    = obj.WInRec;
            wRecOut    = obj.wRecOut;
            save(WFile,'wRec','wRecInit','WInRec','wRecOut')
        end

        function zeroStateRRN(obj)
            % set the network state to zeros

            obj.aRec   = zeros(obj.nRec,1);
            obj.aRecS  = zeros(obj.nRec,1);
            obj.aOut  = zeros(obj.nOut,1);
            obj.aIn   = zeros(obj.nIn,1);
        end

        function randStateRRN(obj)
            % Set the inital state of the network, randomly for  the recurrent units, zeros for In and Out
            obj.aRec   = rand(obj.nRec,1);
            obj.aRecS  = zeros(obj.nRec,1);
            obj.aOut  = zeros(obj.nOut,1);
            obj.aIn   = zeros(obj.nIn,1);
        end

        function RNNStateGPU(obj)
            % set the GPU state variables
            obj.gRec = gpuArray(single(obj.aRec));
            obj.gaRecS = gpuArray(single(obj.aRecS));
        end

        function clearStateVars(obj)
            % clears the state variables: unit activity and training variable
            obj.gRec=[]; obj.gaRecS=[]; obj.gwRec=[]; obj.gWRecOut=[];
            obj.PRecs=[]; obj.cumPreSizeSq=[]; obj.cumPreSizes=[];
            obj.PInds=[]; obj.RecList=[]; obj.PIndsC=[]; obj.PInds=[];
            obj.PRec=[]; obj.PreSyn=[]; obj.cumPreSizesC=[]; obj.aRec=[];
            obj.aRecS=[]; obj.P=[]; obj.aOut=[]; obj.aIn=[];
        end

        function generate_W_P_GPU(obj)
            % organizes the recurrent weights for the GPU kernal
            obj.PIndsC = [];
            obj.cumPreSizesC = [];
            obj.cumPreSizeSq = 0;
            obj.mxPre = 0;
            Ws = [];

			% Sort through recurrent weights and get all synapse locations.
			% Store those locations and their weights in a single array.
			% This is will be passed to the GPU

            for i=1:obj.nRec
                ind = find(obj.wRec(:,i))'; % find presynaptic indices
                obj.PIndsC = [obj.PIndsC ind]; % store all presyn. inds. in single array
                Ws = [Ws obj.wRec(ind,i)']; % store the weights for those synapses
                obj.cumPreSizesC = [obj.cumPreSizesC length(obj.PIndsC)]; % store # of synapses onto each unit
                if length(ind) > obj.mxPre % store largest # of synapses onto single unit
                    obj.mxPre = length(ind);
                end
                obj.cumPreSizeSq = obj.cumPreSizeSq + length(ind)^2; % squared # of synapses. for P-matrix
            end
            obj.PRecs = zeros(1, obj.cumPreSizeSq); % instantiate p-matrix
            obj.cumPreSizeSq=[]; % reset squared synapse size
            stP = 1; % # of items in P matrix

            for i=1:obj.nRec % loop through every unit
                if i > 1 % if after first unit, get number of presynaptic units by taking difference cumPreSizesC between current unit and previous
                    en = obj.cumPreSizesC(i) - obj.cumPreSizesC(i-1);
                else % if first unit, take first item
                    en = obj.cumPreSizesC(i);
                end
                PTmp = eye(en)/obj.alpha; % for each unit's synapses make p-matrix initial state with alpha (forgetting rate of RLS learning rule)
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
            obj.P = gpuArray(single(eye(obj.nRec)/obj.alpha));
            obj.gwRec = gpuArray(single(Ws));
            obj.gWRecOut = gpuArray(single(obj.wRecOut));

            obj.RecList = gpuArray(int32(sort(1:floor(obj.nRec))));
            obj.gRec = gpuArray(single(obj.aRec));
            obj.gaRecS = gpuArray(single(obj.aRecS));
        end

        function generateP_CPU(obj)
            % Gernate the P matrices for CPU training
            for i=1:obj.nRec
                obj.PreSyn(i).ind = find(obj.wRec(:,i));
                obj.PRec(i).P = eye(length(obj.PreSyn(i).ind))/obj.alpha;
            end
            obj.P   = eye(obj.nRec)/obj.alpha;
        end

        function reconWs(obj)
            % reconstruct the recurrent weights from the GPU variables
            Ws = gather(obj.gwRec);
            i1 = 0;
            for i=1:obj.nRec
                i2 = obj.cumPreSizesC(i);
                len = i2-i1;
                obj.wRec(obj.PIndsC(i1+(1:len)),i) = Ws(i1+(1:len))';
                i1 = i2;
            end
            obj.wRecOut=gather(obj.gWRecOut);
        end

        function newState(obj, seed)
            % reseed he rng and store the old seed
            obj.LastSeed = obj.NetworkSeed;
            obj.NetworkSeed = seed;
            rng(obj.NetworkSeed);
        end

        function Seed = getNetworkSeed(obj)
            Seed = obj.Seed;
        end

        function newwRecOut(obj)
            obj.makewRecOut;
        end

        function [DateStr] = getDate(obj)
            DateStr = obj.Date;
        end

        function [Name] = getName(obj)
            Name = ['Seed', num2str(obj.Seed), 'RNN'];
            obj.Name=Name;
        end

        function setRNNTarget(obj, Target)
            obj.Target = Target;
        end

        function setwRec(obj, wRecNew)
            weightDim=size(wRecNew); % check if wRec is correct dimensions
            if ~(numel(weightDim)==2) || ~isempty(find(weightDim~=obj.nRec))
                error('wRecNew is the wrong size! Weights not changed')
            else
                obj.wRec = wRecNew;
                obj.Date = datestr(clock, 30);
            end
        end

        function [targRec] = getRNNTarget(obj)
            targRec = obj.Target;
        end

        function [RecW] = getwRec(obj)
            RecW = obj.wRec;
        end

        function [wRecOut] = getwRecOut(obj)
            wRecOut = obj.wRecOut;
        end

        function [RecWInit] = getwRecInit(obj)
            RecWInit = obj.wRecInit;
        end

        function [WIn] = getWInRec(obj)
            WIn = obj.WInRec;
        end

        function setWInRec(obj, WIn)
            if isequal([obj.nRec,obj.nIn], size(WIn))
                obj.WInRec = WIn;
            else
                error('Error. \nWIn must be size NxIn', class(RNN))
            end
        end

        function [newEx] = setRecS(obj,newaRecS)
            % Sets RNN aRecS to newaRecS, and aRec to tanh(aRecS)

            if isequal(size(newaRecS),[obj.nRec,1])
                obj.aRecS=newaRecS;
                newEx=tanh(obj.aRecS);
                obj.aRec=newEx;
                fprintf([obj.Name,' aRecS set, aRec set to tanh(aRecS)\n'])
            else
                error('newEx must be size [nRec,1]. aRec not set.')
            end
        end
    end

    methods (Access = protected)
        function instantiateRNN(obj)
            obj.makewRec;
            obj.makewRecOut;
            obj.makeWInRec;
            obj.wRecInit = obj.wRec;
        end

        function makewRec(obj)
            WMask = rand(obj.nRec, obj.nRec);
            WMask(WMask<(1-obj.pConn))=0;
            WMask(WMask>0) = 1;
            wRectmp = randn(obj.nRec, obj.nRec)*sqrt(1/(obj.nRec*obj.pConn));
            wRectmp = obj.g*wRectmp.*WMask;
            wRectmp(1:(obj.nRec+1):obj.nRec*obj.nRec)=0;
            obj.wRec = wRectmp;
        end

        function makewRecOut(obj)
            obj.wRecOut = randn(obj.nRec, obj.nOut)*sqrt(1/obj.nRec);
        end

        function makeWInRec(obj)
            obj.WInRec = randn(obj.nRec, obj.nIn);
        end

        function iterateRNN_GPUFull(obj, gwRec, gwRecOut, NoisePlusIn,...
                cumPreSizes, PInds, aRec, aRecS, aOut, t, NumCol,...
                nGridCell, nOut, tau, mxPre)

            mexGPUUpdate(gwRec,...
                gwRecOut,...
                NoisePlusIn,...
                cumPreSizes,...
                PInds,...
                aRec,...
                aRecS,...
                aOut,...
                single(t),...
                single(NumCol),...
                single(nGridCell),...
                single(nOut), ...
                single(tau),...
                single(mxPre));
        end

        function trainRNNFORCE_GPUFull(obj, gEx, error_rec, cumPreSizes,...
                cumPreSizeSq, PInds, PRecs, gwRec, RecList, NumCol,...
                nGridCell, mxPre)
            mexGPUTrain(gEx, ...
                error_rec, ...
                cumPreSizes, ...
                cumPreSizeSq, ...
                PInds, ...
                PRecs, ...
                gwRec, ...
                RecList, ...
                single(NumCol), ...
                single(nGridCell), ...
                single(mxPre), ...
                single(length(RecList)));
        end

    end
end
