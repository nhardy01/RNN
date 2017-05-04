close all;
%% Get parameters
thisRNN = Seed110RNN_20161127T103838;
SF = thisRNN.scalingFactor;
Tau = thisRNN.tau;
G = thisRNN.g;
BL = thisRNN.originalTonicLvl;
Tics = thisRNN.scalingTics;
InfoStr = sprintf('ScaleFact=%.3g G=%.3g Tau=%d BL=%.3g Tic=%.3g\n', ...
    SF, G, Tau, BL, Tics)
%% Test Parameters
TMax = 10000;
TrigStart = 200;
TrigEnd = thisRNN.TrigDur + TrigStart;
AllH = {};

%% test without noise
InterpSS = [thisRNN.ExExTrainTonicStims(end):Tics/6:thisRNN.ExExTrainTonicStims(1)];
%InterpSS = 0.3:0.025:.45;
NumTest = numel(InterpSS);
times = [];
UnitFig = figure; hold on;
InPulses = {};
for testInd = 1:NumTest
    thisSS = InterpSS(testInd);
    numScales = (thisSS-thisRNN.originalTonicLvl)/...
                    thisRNN.scalingTics*thisRNN.scaleDir;
    sigDur = round(thisRNN.TargLen*(1-numScales*thisRNN.scalingFactor));
    times(testInd) = sigDur + TrigEnd;
    InPulse = thisRNN.generateInputPulses([2, 3], [thisRNN.TrigAmp, thisSS],...
        [TrigStart, TrigStart], [TrigEnd, TrigEnd+sigDur] , TMax);
    InPulses{testInd} = InPulse;
    InPlusNoise = thisRNN.generateNoiseInput(InPulse, 0);
    hRNN = zeros(thisRNN.numEx, TMax);
    thisRNN.zeroStateRRN;
    for t = 1:TMax
        In = InPlusNoise(:,t);
        [~, hRNN(:,t)] = thisRNN.IterateRNN_CPU(In);
    end
    figure(UnitFig); plot(hRNN(1,1:times(testInd)));
    title(sprintf([InfoStr, 'Unit 1']));
    AllH{testInd} = hRNN;
    figure
    imagesc(hRNN); title(sprintf('SpeedSignal = %.3g', thisSS)); drawnow;
end
targs = thisRNN.getRNNTarget;
% Targ1F = figure; hold on;
% plot(AllH{1}(1,100:end)); plot(targs{3}(1,:),'g');
Targ2F = figure; hold on;
plot(AllH{1}(1,100:end)); %plot(targs{2}(1,:),'g');
Targ3F = figure; hold on;
plot(AllH{end}(1,100:end)); %plot(targs{1}(1,:),'g');

refEnd = round(times(1))+200;
refTraj = AllH{1}(:, 1:refEnd);
Dists = zeros(NumTest, round(max(times))+200);
Times = zeros(NumTest, round(max(times))+200);
for testInd = 1:NumTest
    testEnd = round(times(testInd))+200;
    D = dist([refTraj, AllH{testInd}(:, 1:testEnd)]);
    [minD, minInd] = min(D(refEnd+1:end, 1:refEnd));
    Dists(testInd,1:size(minD,2)) = minD;
    Times(testInd,1:size(minD,2)) = minInd;
end

figure; plot(Dists'); title(sprintf([InfoStr, 'Min. Distance from Longest']));
figure; plot(Times'); title(sprintf([InfoStr, 'Time of Min. Dist.']));