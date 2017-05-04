function TestRNN(RNN,SaveDir,NoiseAmp,TrialsPerStim,HExSaveTrials,Debug,CueIn,SpeedIn,InterpSS)
%% Check and organize inputs
narginchk(5,9);
switch nargin
    case 5
        Debug=false;
        CueIn=RNN.CueIn;
        SpeedIn=RNN.SpeedIn;
        InterpSS=[0.0750,0.099,0.15,0.225,0.3];% to replicate psychophysics speeds, assumes these standard speeds
    case 6
        CueIn=RNN.CueIn;
        SpeedIn=RNN.SpeedIn;
        InterpSS=[0.0750,0.099,0.15,0.225,0.3];% to replicate psychophysics speeds, assumes these standard speeds
    case 7
        SpeedIn=RNN.SpeedIn;
        InterpSS=[0.0750,0.099,0.15,0.225,0.3];% to replicate psychophysics speeds, assumes these standard speeds
    case 8
        InterpSS=[0.0750,0.099,0.15,0.225,0.3];% to replicate psychophysics speeds, assumes these standard speeds
end
%% Parameters
TrigStart=500;
Threshold=0.2;
peekU=randi(RNN.numEx,1);
XDurr=0.15/min(InterpSS)+1; % normalize total length, assuming 1x speed is 4000 ms
TrigDur=RNN.TrigDur;
TrigEnd=TrigDur+TrigStart;
TotT=round(4000*XDurr+1000+TrigEnd);
NumSpeeds=numel(InterpSS);
%% Define variables
AllHEx=cell(NumSpeeds,HExSaveTrials,numel(CueIn)); % only store the number of desired hEx trials
PeakTimes=zeros(NumSpeeds,TrialsPerStim,5,numel(CueIn));
AllHIn=cell(NumSpeeds,TrialsPerStim,numel(CueIn));
AllHOut=cell(NumSpeeds,TrialsPerStim,numel(CueIn));
StimOff=zeros(NumSpeeds,TrialsPerStim,numel(CueIn));
outFigT = figure; hold on;
o1h = subplot(size(InterpSS,2),1,1);
recFigT = figure; hold on;
r1h = subplot(size(InterpSS,2),1,1);
FullRecFig=figure;
if Debug; viewHitFig=figure; end;
%% Start simulations
TestSeed=randi(1000); % make sure the testing seed is different than the traing seed
while TestSeed==RNN.getNetworkSeed
    TestSeed=randi(1000);
end
RNN.newState(TestSeed);
for repNum = 1:TrialsPerStim
    for thisSpeedStim=1:NumSpeeds
        for cueInd=1:numel(CueIn)
            thisCue=CueIn(cueInd);
            thisSS=InterpSS(thisSpeedStim);
            InPulse=RNN.generateInputPulses([thisCue,SpeedIn],[RNN.TrigAmp,thisSS],...
                [TrigStart,TrigStart],[TrigEnd,TotT-TrigEnd],TotT);
            hEx=zeros(RNN.numEx,TotT);
            hOut=zeros(RNN.numOut,TotT);
            hIn=zeros(RNN.numIn,TotT);
            RNN.randStateRRN;
            count=0; lastHit=[];
            for t = 1:TotT
                In=InPulse(:,t);
                if ~isempty(lastHit) && t>(lastHit*1.2)
                    In(3) = 0; % remove for constant input dur accross speeds
                    if ~StimOff(thisSpeedStim,repNum,cueInd)
                        StimOff(thisSpeedStim,repNum,cueInd)=t;
                    end
                end
                hIn(:,t)=In;
                InNoise=RNN.getWInEx*In+randn(RNN.numEx,1)*RNN.innateNoiseLvl*NoiseAmp;
                [~,hEx(:,t)]=RNN.IterateRNN_CPU(InNoise);
                hOut(:,t)=RNN.IterateOutCPU;
                if t>TrigEnd+1300 && count<5
                    smthH=smooth(hOut(TrigEnd+1:t),150);
                    [pks,locs]=findpeaks(smthH(1:end-100),...
                        'MinPeakDistance',125,...
                        'MinPeakProminence',0.2,...
                        'MinPeakHeight',Threshold);
                    count=numel(locs);
                elseif count==5 && isempty(lastHit)
                    lastHit=locs(5)+TrigEnd;
                    PeakTimes(thisSpeedStim,repNum,:,cueInd)=locs+TrigEnd;
                end
            end
            if Debug
                %%% For debugging and parameter optimization
                figure(viewHitFig); clf; hold on;
                plot(hOut(TrigEnd+1:t));plot(smthH,'linewidth',2);
                plot(locs,pks,'k.','markersize',20);
                drawnow; beep;
                waitforbuttonpress;
            end
            %%%% Store data
            AllHOut{thisSpeedStim,repNum,cueInd}=hOut;
            AllHIn{thisSpeedStim,repNum,cueInd}=hIn;
            if repNum<=HExSaveTrials % only save desired # of hEx trials
                AllHEx{thisSpeedStim,repNum,cueInd}=hEx;
            end
            %%% remove trials that did not reach 5 hits
            if ~isempty(find(PeakTimes(thisSpeedStim,repNum,:,cueInd)==0,1))
                PeakTimes(thisSpeedStim,repNum,:,cueInd)=NaN;
            end
            %%% Plot output
            figure(recFigT); subplot(NumSpeeds,1,thisSpeedStim); hold on;
            plot(hIn'); plot(hEx(peekU,:)); ylim([-1,1]); drawnow;
            figure(outFigT); subplot(NumSpeeds,1,thisSpeedStim); hold on;
            plot(hOut'); plot(hIn'); ylim([-.5 1]);
            thisTrialPeak=squeeze(PeakTimes(thisSpeedStim,repNum,:,cueInd));
            if isempty(find(isnan(thisTrialPeak),1))
                plot(thisTrialPeak,hOut(thisTrialPeak),'.k',...
                    'MarkerSize',15)
            end
            title(o1h,sprintf('Out Test Noise%g',RNN.innateNoiseLvl));
            title(r1h,sprintf('RNN Test Noise%g',RNN.innateNoiseLvl));
            figure(FullRecFig); clf; imagesc(hEx);
            drawnow;
        end
    end
end
%% Process hit time stats
MnHit=squeeze(nanmean(PeakTimes,2));
StdHit=squeeze(nanstd(PeakTimes,[],2));
VarHit=squeeze(nanvar(PeakTimes,[],2));
%% Save data
if ~exist(SaveDir,'dir')
    mkdir(SaveDir)
end
saveName=[RNN.getName,sprintf('_TestActivity_ExtrapOnly',NoiseAmp)];
save(fullfile(SaveDir,saveName),'AllHOut','AllHIn','AllHEx','TrigStart',...
    'NoiseAmp','TestSeed','PeakTimes','StimOff','InterpSS','TrigDur',...
    'MnHit','StdHit','VarHit','-v7.3')
end
