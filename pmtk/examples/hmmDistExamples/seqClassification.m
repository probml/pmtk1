%% Sequence Classification
%#broken
load data45; 
nstates = 5;
%% 
%Initial Guesses for EM
pi0 = [1,0,0,0,0];
transmat0 = normalize(diag(ones(nstates,1)) + diag(ones(nstates-1,1),1),2);
%%
condDensity = HmmDist('nstates',5,'observationModel',MvnDist());
%%
model = GenerativeClassifierDist('classConditionals',condDensity,'nclasses',2,'classSupport',4:5);
%%
trainingData = {train4,train5};
trainingLabels = [4,5];
fitOptions = {'transitionMatrix0',transmat0,'pi0',pi0};
%%
model = fit(model,'observations',trainingData,'labels',trainingLabels,'fitOptions',fitOptions);
%%
pred = predict(model,test45);
yhat = mode(pred);
nerrors = sum(yhat ~= labels');
display(nerrors);