%%  A Simple Test of the HmmDist Class
%#testPMTK
%% Discrete Observations
setSeed(0);
trueObsModel = {DiscreteDist('mu',ones(6,1)./6       ,'support',1:6)
    DiscreteDist('mu',[ones(5,1)./10;0.5],'support',1:6)};

trueTransDist = DiscreteDist('mu',[0.8,0.2;0.3,0.70]','support',1:2);
trueStartDist = DiscreteDist('mu',[0.5,0.5]','support',1:2);
trueModel = HmmDist('startDist'     ,trueStartDist,...
    'transitionDist',trueTransDist,...
    'emissionDist'  ,trueObsModel);

nsamples = 20; length1 = 13; length2 = 30;
[observed1,hidden1] = sample(trueModel,nsamples/2,length1);
[observed2,hidden2] = sample(trueModel,nsamples/2,length2);
observed = [num2cell(squeeze(observed1),1)';num2cell(squeeze(observed2),1)'];

model = HmmDist('emissionDist',DiscreteDist('support',1:6),'nstates',2);
model = fit(model,'data',observed);

model = condition(model,'Y',observed{1}');
postSample = mode(samplePost(model,1000),2)'
viterbi = mode(model)
maxmarg = maxidx(marginal(model,':'))
%% MVN Observations
trueObsModel = {MvnDist(zeros(1,10),randpd(10));MvnDist(ones(1,10),randpd(10))};
trueTransDist = DiscreteDist('mu',[0.8,0.2;0.1,0.90]','support',1:2);
trueStartDist = DiscreteDist('mu',[0.5;0.5],'support',1:2);
trueModel = HmmDist('startDist'     ,trueStartDist,...
    'transitionDist',trueTransDist,...
    'emissionDist'  ,trueObsModel);
nsamples = 50; length = 5;
[observed,trueHidden] = sample(trueModel,nsamples,length);
model = HmmDist('emissionDist',MvnDist(),'nstates',3);
model = fit(model,'data',observed);
%% MvnMixDist Observations

nstates = 5; d = 2; nmixcomps = 2;
emissionDist = cell(5,1);
for i=1:nstates
    emissionDist{i} = mkRndParams(MvnMixDist('nrestarts',5,'verbose',false),d,nmixcomps);
end
pi = DiscreteDist('mu',normalize(ones(nstates,1)));
A = DiscreteDist('mu',normalize(rand(nstates),1));
trueModel = HmmDist('startDist',pi,'transitionDist',A,'emissionDist',emissionDist,'nstates',nstates);
[observed,hidden] = sample(trueModel,1,500);
model = HmmDist('emissionDist',MvnMixDist('nmixtures',nmixcomps,'verbose',false,'nrestarts',1),'nstates',nstates);
model = fit(model,'data',observed,'maxIter',20);
%% DiscreteMixDist Observations

nstates = 5;  nmixcomps = 2; d = 3;
emissionDist = cell(5,1);
for i=1:nstates
    emissionDist{i} = mkRndParams(DiscreteMixDist('nmixtures',nmixcomps),d,nmixcomps);
end
pi = DiscreteDist('mu',normalize(rand(nstates,1)));
A = DiscreteDist('mu',normalize(rand(nstates),1));
trueModel = HmmDist('startDist',pi,'transitionDist',A,'emissionDist',emissionDist,'nstates',nstates);
[observed,hidden] = sample(trueModel,1,500);
model = HmmDist('emissionDist',DiscreteMixDist('nmixtures',nmixcomps,'verbose',false),'nstates',nstates);
model = fit(model,'data',observed,'maxIter',20);