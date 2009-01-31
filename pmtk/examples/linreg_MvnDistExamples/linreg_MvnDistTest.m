%% Simple Test of Linreg_MvnDist
load prostate;
lambda = 0.05;
T = ChainTransformer({StandardizeTransformer(false),AddOnesTransformer()});
%XX = train(T, X);  d = size(XX,2);
%prior = makeSphericalPrior(d, lambda, addOffset(T), 'mvnig');
%model = Linreg_MvnInvGammaDist('wSigmaDist', prior, 'transformer',T);
model = Linreg_MvnDist('transformer',T, 'priorStrength', lambda,'sigma2',1);
model = fit(model,'X',Xtrain,'y',ytrain);
yp = predict(model,Xtest);
err = mse(ytest, mode(yp))