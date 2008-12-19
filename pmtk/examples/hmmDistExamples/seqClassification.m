%% Sequence Classification
load data45; 
nstates = 5;
%% 
%Initial Guesses for EM
pi0 = [1,0,0,0,0];
transmat0 = normalize(diag(ones(nstates,1)) + diag(ones(nstates-1,1),1),2);
startDist = DiscreteDist('mu',pi0','support',1:5);
transDist = DiscreteDist('mu',transmat0','support',1:5);
%%
classConditionals = copy(HmmDist('nstates',5,'emissionDist',MvnDist(),'transitionDist',transDist,'startDist',startDist),2,1);
%%
model = GenerativeClassifierDist('classConditionals',classConditionals,'classPrior',DiscreteDist('mu',[0.5,0.5],'support',4:5));
%%
trainingData = {train4;train5};
trainingLabels = [4,5];
%%
model = fit(model,'X',trainingData,'y',trainingLabels);
%%
pred = predict(model,test45');
yhat = mode(pred);
nerrors = sum(yhat ~= labels');
display(nerrors);