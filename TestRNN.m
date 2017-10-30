function TestRNN(Net,SaveDir,NoiseAmp,TrialsPerStim,HExSaveTrials,Debug,CueIn,SpeedIn,InterpSS)
%% Check and organize inputs

%% Parameters
TrigStart=500;
Threshold=0.2;
peekU=randi(Net.nRec,1);
TrigDur=Net.tau*5;
TrigAmp = 5;
TrigEnd=TrigDur+TrigStart;
TotT=9000+TrigEnd;
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
while TestSeed==Net.getNetworkSeed
    TestSeed=randi(1000);
end
Net.newState(TestSeed);
for repNum = 1:TrialsPerStim
    for thisSpeedStim=1:NumSpeeds
        for cueInd=1:numel(CueIn)
            thisCue=CueIn(cueInd);
            thisSS=InterpSS(thisSpeedStim);
            InPulse=Net.generateInputPulses([thisCue,SpeedIn],[TrigAmp,thisSS],...
                [TrigStart,TrigStart],[TrigEnd,TotT-TrigEnd],TotT);
            hEx=zeros(Net.nRec,TotT);
            hOut=zeros(Net.nOut,TotT);
            hIn=zeros(Net.nIn,TotT);
            Net.randStateRRN;
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
                InNoise=Net.getWInRec*In+randn(Net.nRec,1)*NoiseAmp;
                [~,hEx(:,t)]=Net.IterateRNN_CPU(InNoise);
                hOut(:,t)=Net.IterateOutCPU;
                if t>TrigEnd+1300 && count<5
                    smthH=smooth(hOut(TrigEnd+1:t),150);
                    [pks,locs]=findpeaks(smthH(1:end-100),...
                        'MinPeakDistance',125,...
                        'MinPeakHeight',Threshold);
                    % 'MinPeakProminence',0.2,...
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
            title(o1h,sprintf('Out Test Noise%g',NoiseAmp));
            title(r1h,sprintf('Net Test Noise%g',NoiseAmp));
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
saveName=[Net.getName,sprintf('_TestActivity_ExtrapOnly',NoiseAmp)];
save(fullfile(SaveDir,saveName),'AllHOut','AllHIn','AllHEx','TrigStart',...
    'NoiseAmp','TestSeed','PeakTimes','StimOff','InterpSS','TrigDur',...
    'MnHit','StdHit','VarHit','-v7.3')
end
