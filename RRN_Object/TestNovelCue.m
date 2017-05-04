function TestNovelCue(RNN,SaveDir,NoiseAmp,TrialsPerStim,HExSaveTrials,NumIn,Debug)
%{
Wrapper for TestRNN function. Adds new inputs to the RNN and tests them and
the original inputs. No Speed Signal delivered.
%}
narginchk(6,7)
if nargin==6; Debug=false; end        
NewWIn=randn(RNN.numEx,NumIn);
oldNumIn=RNN.numIn;
RNN.numIn=oldNumIn+NumIn;
RNN.setWInEx([RNN.getWInEx,NewWIn])
WDir=fullfile(SaveDir,'NovelWeights');
if ~exist(WDir,'dir'); mkdir(WDir); end
RNN.saveWeights(WDir)
TestRNN(RNN,WDir,NoiseAmp,TrialsPerStim,HExSaveTrials,Debug,...
    [RNN.CueIn,[1:NumIn]+oldNumIn],RNN.SpeedIn,[0,RNN.ExExTrainTonicStims(1)])
end