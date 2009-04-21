%% Train an HMM on the word "four" and then find the Viterbi parse of a test signal into phones
%#broken
function wordSegmentation()

setSeed(0);
if(~exist('data45.mat','file'))
  error('Please download data45.mat from www.cs.ubc.ca/~murphyk/pmtk and save it in the data directory');
end
helper(false);
helper(true);
end

function helper(useMix)
load data45;
nstates = 5; d = 13; nmixcomps = 3;
startDist = DiscreteDist('-T',[1,0,0,0,0]','-support',1:5);
transmat0 = normalize(diag(ones(nstates,1)) + diag(ones(nstates-1,1),1),2);
transDist = DiscreteDist('-T',transmat0','-support',1:5);
emissionDist = cell(5,1);
for i=1:nstates
  if useMix
    emissionDist{i} = mkRndParams(MvnMixDist('nrestarts',1,'verbose',false),d,nmixcomps);
  else
    emissionDist{i} = mkRndParams(MvnDist(),d);
  end
end            
model4 = HmmDist('startDist',startDist,'transitionDist',transDist,'emissionDist',emissionDist,'nstates',nstates);
model4 = fit(model4,'data',train4,'maxIter',20);
if(exist('specgram','file')) % need signal processing toolbox to view spectogram
  figure;
  subplot(2,2,1);
  specgram(signal1);
  subplot(2,2,2)
  specgram(signal2);
  subplot(2,2,3);
  model4 = condition(model4,'Y',mfcc1);
  plot(mode(model4));
  set(gca,'YTick',1:5);
  subplot(2,2,4);
  model4 = condition(model4,'Y',mfcc2);
  plot(mode(model4));
  set(gca,'YTick',1:5);
  maximizeFigure;
end
end

