%% LinregDist on the Prostate Data Set
load prostate;
lambda = 0.05;
sigma2 = 0.5;
T = ChainTransformer({StandardizeTransformer(false),AddOnesTransformer()});
model = LinregDist('transformer',T);
%% L1 shooting
modelL1shooting = fit(model,'X',Xtrain,'y',ytrain,'prior','l1','lambda',lambda,'algorithm','shooting');
yp = predict(modelL1shooting,Xtest);
L1shootingErr = mse(ytest,mode(yp))
%% L1 lars
modelL1lars = fit(model,'X',Xtrain,'y',ytrain,'prior','l1','lambda',lambda,'algorithm','lars');
yp = predict(modelL1lars,Xtest);
L1larsErr = mse(ytest,mode(yp))
%% L2 QR
modelL2qr = fit(model,'X',Xtrain,'y',ytrain,'prior','l2','lambda',lambda,'algorithm','ridgeQR');
yp = predict(modelL2qr,Xtest);
L2qrErr = mse(ytest,mode(yp))
%% L2 SVD
modelL2svd = fit(model,'X',Xtrain,'y',ytrain,'prior','l2','lambda',lambda,'algorithm','ridgeSVD');
yp = predict(modelL2svd,Xtest);
L2svdErr = mse(ytest,mode(yp))
%% Elastic Net, lars
modelElasticLars = fit(model,'X',Xtrain,'y',ytrain,'prior','l1l2','lambda',[lambda,lambda],'algorithm','lars');
yp = predict(modelElasticLars,Xtest);
elasticLarsErr = mse(ytest,mode(yp))
%% Elastic Net shooting
modelElasticShooting = fit(model,'X',Xtrain,'y',ytrain,'prior','l1l2','lambda',[lambda,lambda],'algorithm','shooting');
yp = predict(modelElasticShooting,Xtest);
elasticShootingErr = mse(ytest,mode(yp))
%% MLE
modelMLE = fit(model,'X',Xtrain,'y',ytrain);
yp = predict(modelMLE,Xtest);
mleErr = mse(ytest,mode(yp))