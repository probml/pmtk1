%% Sequence Classification
setSeed(10);
load data45; 
%%
nstates = 5;
pi0 = [1,0,0,0,0];
transmat0 = normalize(diag(ones(nstates,1)) + diag(ones(nstates-1,1),1),2);
%%
condDensity = HmmDist('nstates',5,'observationModel',MvnDist());
%%
model = GenerativeClassifierDist('classConditionals',condDensity,'nclasses',2,'classSupport',4:5);
%%
obsData = {train4,train5};
hidData = [4,5];
fitOptions = {'transitionMatrix0',transmat0,'pi0',pi0,'maxIter',5};
%%
model = fit(model,'dataObs',obsData,'dataHid',hidData,'fitOptions',fitOptions);
%%
pred = predict(model,test45);
yhat = mode(pred);
nerrors = sum(yhat ~= labels');
display(nerrors);