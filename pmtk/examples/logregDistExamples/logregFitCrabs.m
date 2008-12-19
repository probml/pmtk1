%% Logistic Regression Crabs Data
% Here we fit Logistic Regression to the crabs data set using various
% approximations and compare the results. 
%% Setup
[Xtrain, ytrain, Xtest, ytest] = makeCrabs;
sigma2 = 32/5;  lambda = 1e-3;
T = ChainTransformer({StandardizeTransformer(false), KernelTransformer('rbf', sigma2)});
%% MLE
m = LogregDist('nclasses',2, 'transformer', T);
m = fit(m, 'X', Xtrain, 'y', ytrain);
Pmle   = predict(m,Xtest);
%% MAP
m = LogregDist('nclasses',2, 'transformer', T);
m = fit(m, 'X', Xtrain, 'y', ytrain, 'priorStrength', lambda,'prior','l2');
Pmap   = predict(m,Xtest);
%% Bayesian (Laplace / Integral)
m = Logreg_MvnDist('nclasses',2,'transformer',T);
m = fit(m, 'X', Xtrain, 'y', ytrain, 'priorStrength', lambda,'infMethod','laplace');
PbayesLAint = predict(m,Xtest,'method','integral');
%% Bayesian (Laplace / MC)
m = Logreg_MvnDist('nclasses',2,'transformer',T);
m = fit(m, 'X', Xtrain, 'y', ytrain, 'priorStrength', lambda,'infMethod','laplace');
PbayesLAmc = predict(m,Xtest,'method','mc');
%% Bayesian (MH / MC)
m = Logreg_MvnDist('nclasses',2,'transformer',T);
m = fit(m, 'X', Xtrain, 'y', ytrain, 'priorStrength', lambda,'infMethod','mh');
PbayesMHmc = predict(m,Xtest,'method','mc');
%%
nerrsMLE = sum(mode(Pmle) ~= ytest)
nerrsMAP = sum(mode(Pmap) ~= ytest)                                        
nerrsPbayesLAint = sum(mode(PbayesLAint) ~= ytest)
nerrsPbayesLAmc  = sum(mode(PbayesLAmc)  ~= ytest)
nerrsPbayesMHmc  = sum(mode(PbayesMHmc) ~= ytest)

