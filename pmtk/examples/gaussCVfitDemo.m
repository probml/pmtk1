%% Estimate mu/sigma by Cross Validation
%% Sample
%#broken - uses old ModelSelection Class
setSeed(1);
mu = 0; sigma = 2;
mtrue = GaussDist(mu, sigma);
ntrain = 300;
Xtrain = sample(mtrue, ntrain);
%% Model Selection
muLambda = -10:0.5:10;
sigmaLambda = 0.1:0.1:4;
for l=1:length(lambda)

end
modelSpace = ModelDist.makeModelSpace(-10:0.5:10,0.1:0.1:4);
scoreFunction = @(md,model)cvScore(GaussDist(model{1},model{2}),md.Xdata,'clamp',true);
md = fit(ModelDist('scoreFunction'     ,scoreFunction,...
                   'Xdata'             ,Xtrain       ,...
                   'models'            ,modelSpace   ,...
                   'scoreTransformer'  ,@(x)-exp(x)));
%% CV Best Guess
% We take the map estimate of the distribution over models. 
cvMAPmodel  = GaussDist(md.mapEstimate{1},md.mapEstimate{2})
%% MLE
% Here we compare the CV selected model to the MLE. 
fitMLEmodel = fit(GaussDist(),'data',Xtrain)

