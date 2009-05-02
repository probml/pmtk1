%% Logistic Regression on binary Crabs Data
% Here we fit (kernelized) Logistic Regression to the crabs data set using various
% approximations and compare the results. 
%#testPMTK
%% Setup
[Xtrain, ytrain, Xtest, ytest] = makeCrabs; % y in {1,2}
Dtrain = DataTable(Xtrain, ytrain);
Dtest = DataTable(Xtest, ytest);
sigma2 = 32/5;  lambda = 1e-3;
T = ChainTransformer({StandardizeTransformer(false), KernelTransformer('rbf', sigma2)});
%% MLE
mMLE = Logreg('-nclasses',2, '-transformer', T);
mMLE = fit(mMLE, Dtrain);
ymle   = predict(mMLE,Xtest);
%% MAP
mMAP = LogregL2('-nclasses',2, '-transformer', T,'-lambda',lambda);
mMAP = fit(mMAP, Dtrain);
[ymap,pmap]   = predict(mMAP,Xtest);

%% MAP binary
mMAP2 = LogregBinaryL2('-transformer', T,'-lambda',lambda,'-optMethod','lbfgs');
[mMAP2, out2] = fit(mMAP2, Dtrain);
[ymap2,pmap2]   = predict(mMAP2,Xtest);
ndiff = sum(ymap ~= ymap2);
assert(ndiff==0)
assert(approxeq(abs(mMAP.w), abs(mMAP2.w)))
assert(approxeq(abs(mMAP.w0), abs(mMAP2.w0))) 
% multinomial in the 2 class case should give same results as binary logreg


mMAP3 = LogregBinaryL2('-transformer', T,'-lambda',lambda,'-optMethod', 'irls');
[mMAP3, out3] = fit(mMAP3, Dtrain);
ymap3   = predict(mMAP3,Xtest);
ndiff = sum(ymap3~=ymap2);
assert(ndiff==0);
tol = 0.5;
assert(approxeq(abs(mMAP3.w), abs(mMAP2.w), tol))
assert(approxeq(abs(mMAP3.w0), abs(mMAP2.w0), tol))
% The LBFGS optimizer should give same results as IRLS
% although our IRLS implementation is not very sophisticated.

%% Bayesian (Laplace / Integral)
mLap = LogregBinaryLaplace('-transformer',T,'-lambda',lambda);
mLap = fit(mLap, Dtrain);
ylap = predict(mLap,Xtest);
%%
nerrsMLE = sum(ymle ~= ytest)
nerrsMAP = sum(ymap ~= ytest)  
nerrsMAP2 = sum(ymap2 ~= ytest)
nerrsLap = sum(ylap ~= ytest) 

%{
nr = 2; nc = 2;
figure;
subplot(nr,nc,1); stem(ytest);title('test')
subplot(nr,nc,2); stem(ymap);title('map')
subplot(nr,nc,3); stem(ymap2);title('map2')
subplot(nr,nc,4); stem(ymap3);title('map3')
%}

