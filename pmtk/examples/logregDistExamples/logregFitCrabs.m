%% Logistic Regression Crabs Data
% Here we fit an L2 regularized logistic regression model to the crabs
% data set and predict using three methods: MAP plugin approx, Monte
% Carlo approx, and using a closed form approximation to the posterior
% predictive.
[Xtrain, ytrain, Xtest, ytest] = makeCrabs;
sigma2 = 32/5;
T = ChainTransformer({StandardizeTransformer(false), KernelTransformer('rbf', sigma2)});
m = LogregDist('nclasses',2, 'transformer', T);
lambda = 1e-3;
m = fit(m, 'X', Xtrain, 'y', ytrain, 'lambda', lambda,'prior','l2','method','bayesian');
Pmap   = predict(m,'X',Xtest,'method','plugin');
Pmc    = predict(m,'X',Xtest,'method','mc');
Pexact = predict(m,'X',Xtest,'method','integral');
nerrsMAP   = sum(mode(Pmap)' ~= ytest)                                           %#ok
nerrsMC    = sum(mode(Pmc)' ~= ytest)                                       %#ok
nerrsExact = sum(mode(Pexact)' ~= ytest)