%% Simple Test of Linreg_MvnInvGammaDist
load prostate;
lambda = 0.05;
T = ChainTransformer({StandardizeTransformer(false),AddOnesTransformer()});
%XX = train(T, X);  d = size(XX,2);
%prior = makeSphericalPrior(d, lambda, addOffset(T), 'mvnig');
%model = Linreg_MvnInvGammaDist('wSigmaDist', prior, 'transformer',T);
model = Linreg_MvnInvGammaDist('transformer',T);
model = fit(model,'X',Xtrain,'y',ytrain, 'priorStrength',lambda);
yp = predict(model,Xtest);
err = mse(ytest, mode(yp))