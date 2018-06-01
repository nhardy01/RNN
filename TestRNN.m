function TestRNN(Net, T, varargin)
%% Check and organize inputs
% amp_noise,n_trials_stim,b_debug,ix_cue_unit,ix_speed_unit,amp_speed
p = inputParser;

p.addRequired('Net', @(x) (isa(x, 'RNN') ))
p.addRequired('T',   @isnumeric)

p.addParameter('SaveDir',               pwd,    @isdir)
p.addParameter('Repetitions',           1,      @isnumeric)
p.addParameter('NRecTrialSave',         0,      @isnumeric)
p.addParameter('Debug',                 false,  @islogical)
p.addParameter('CueUnit',               2,      @isnumeric)
p.addParameter('SpeedUnit',             1,      @isnumeric)
p.addParameter('SpeedAmplitude',        0,      @isnumeric)
p.addParameter('CueAmplitude',          5,      @isnumeric)
p.addParameter('NoiseAmplitude',        0,      @isnumeric)
p.addParameter('StartTime',             NaN,    @isnumeric)
p.addParameter('CueDur',                NaN,    @isnumeric)
p.addParameter('PeakDetectTheshold',    0.2,    @isnumeric)

p.parse(Net, T, varargin{:})

%% Set Parameters
t_start         = p.Results.StartTime;
if isnan(p.Results.StartTime)
    t_start     = T/10;
end
t_cue_dur       = Net.tau * 5;
if ~isnan(p.Results.CueDur)
    t_cue_dur   = p.Results.CueDur;
end
t_cue_end       = t_cue_dur + t_start;
assert(T > t_cue_end, 'Total trial duration is less than start time')

amp_cue         = p.Results.CueAmplitude;
amp_peak_thresh = p.Results.PeakDetectTheshold;
amp_noise       = p.Results.NoiseAmplitude;
amp_speed       = p.Results.SpeedAmplitude;

ix_cue_unit     = p.Results.CueUnit;
ix_speed_unit   = p.Results.SpeedUnit;
ix_view_unit    = randi(Net.nRec,1);

n_speed         = numel(amp_speed);
n_rec_save      = p.Results.NRecTrialSave;
n_trials_stim   = p.Results.Repetitions;
n_cue           = numel(ix_cue_unit);

seed_test = randi(1000); % make sure the testing seed is different than the training seed
while seed_test == Net.getNetworkSeed
    seed_test = randi(1000);
end
Net.newState(seed_test);

dir_save        = p.Results.SaveDir;
b_debug         = p.Results.Debug;

%% Define variables
AllHEx    = cell( n_speed, n_rec_save,    n_cue); % only store the number of desired d_Rec trials
AllHIn    = cell( n_speed, n_trials_stim, n_cue);
AllHOut   = cell( n_speed, n_trials_stim, n_cue);
StimOff   = zeros(n_speed, n_trials_stim, n_cue);
PeakTimes = zeros(n_speed, n_trials_stim, 5,     n_cue);

%% Setup figures
h.Out                  = figure; hold on;
h.Out_sub_1            = subplot(max(n_speed,1),1,1);
h.Rec_U                = figure; hold on;
h.Rec_U_sub_1          = subplot(max(n_speed,1),1,1);
h.Rec_Full             = figure;
if b_debug; h.View_hit = figure; end
%% Start simulations
for i_rep = 1:n_trials_stim
    for i_speed = 1:n_speed
        for i_cue = 1:n_cue
            c_cue = ix_cue_unit(i_cue);
            c_speed  = amp_speed(i_speed);
            amp_in_all = Net.generateInputPulses(   [c_cue,     ix_speed_unit], ...
                                                    [amp_cue,   c_speed], ...
                                                    [t_start,   t_start], ...
                                                    [t_cue_end, T-t_cue_end], ...
                                                    T);
            
            d_Rec       = zeros(Net.nRec, T);
            d_Out       = zeros(Net.nOut, T);
            d_In        = zeros(Net.nIn, T);

            count_hits = 0;
            t_last_hit = [];
            Net.randStateRRN;
            for t = 1:T
                In = amp_in_all(:, t);
                if ~isempty(t_last_hit) && t>(t_last_hit*1.2)
                    In(ix_speed_unit) = 0; % remove for constant input dur accross speeds
                    if StimOff(i_speed, i_rep, i_cue) == 0
                        StimOff(i_speed, i_rep, i_cue) = t;
                    end
                end
                d_In(:,t)      = In;
                In_noise       = Net.getWInRec * In + randn(Net.nRec,1) * amp_noise;
                [~,d_Rec(:,t)] = Net.IterateRNN_CPU(In_noise);
                d_Out(:,t)     = Net.IterateOutCPU;
                if t > (t_cue_end + T/4) && count_hits < 5
                    d_Out_smooth = smooth(d_Out(t_cue_end+1:t),150);
                    [pks,locs]   = findpeaks(d_Out_smooth(1:end-100),...
                                            'MinPeakDistance',125,...
                                            'MinPeakHeight',amp_peak_thresh);
                    count_hits = numel(locs);
                elseif count_hits == 5 && isempty(t_last_hit)
                    t_last_hit = locs(5) + t_cue_end;
                    PeakTimes(i_speed,i_rep,:,i_cue) = locs + t_cue_end;
                end
            end
            if b_debug
                %%% For b_debugging and parameter optimization
                figure(h.View_hit); clf; hold on;
                plot(d_Out(t_cue_end+1:t));
                plot(d_Out_smooth,'linewidth',2);
                plot(locs,pks,'k.','markersize',20);
                drawnow; beep;
                waitforbuttonpress;
            end
            %%%% Store data
            AllHOut{i_speed,i_rep,i_cue} = d_Out;
            AllHIn {i_speed,i_rep,i_cue} = d_In;
            if i_rep <= n_rec_save % only save desired # of d_Rec trials
                AllHEx{i_speed,i_rep,i_cue} = d_Rec;
            end
            %%% remove trials that did not reach 5 hits
            if ~isempty(find(PeakTimes(i_speed,i_rep,:,i_cue)==0,1))
                PeakTimes(i_speed,i_rep,:,i_cue) = NaN;
            end
            %%% Plot output
            figure(h.Rec_U); subplot(n_speed,1,i_speed); hold on;
            plot(d_In');
            plot(d_Rec(ix_view_unit,:));
            ylim([-1,1]); drawnow;
            figure(h.Out); subplot(n_speed,1,i_speed); hold on;
            plot(d_Out');
            plot(d_In');
            ylim([-.5 1]);
            c_t_peaks = squeeze(PeakTimes(i_speed,i_rep,:,i_cue));
            if all(~isnan(c_t_peaks))
                plot(c_t_peaks,d_Out(c_t_peaks),'.k',...
                    'MarkerSize',15)
            end
            title(h.Out_sub_1,sprintf('Out Test, Noise=%g',amp_noise));
            title(h.Rec_U_sub_1,sprintf('RNN Test, Noise=%g',amp_noise));
            figure(h.Rec_Full); clf;
            imagesc(d_Rec);
            drawnow;
        end
    end
end
%% Process hit time stats
MnHit  = squeeze(nanmean(PeakTimes,2));
StdHit = squeeze(nanstd(PeakTimes,[],2));
VarHit = squeeze(nanvar(PeakTimes,[],2));
%% Save data
if ~exist(dir_save,'dir')
    mkdir(dir_save)
end
saveName=[Net.getName,sprintf('_TestActivity_Noise%.3g',amp_noise * 100)];
save(fullfile(dir_save,saveName),'AllHOut','AllHIn','AllHEx','t_start',...
    'amp_noise','seed_test','PeakTimes','StimOff','amp_speed','t_cue_dur',...
    'MnHit','StdHit','VarHit','-v7.3')
end
