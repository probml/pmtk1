%% Comparing Logistic Regression to SVM
% We attempt to reproduce many of the results in Sparse Multinomial Logistic
% Regression: Fast Algorithms and Generalization Bounds Krishnapuram et al,
% (2005)
%
%%
%
runFull = false;
% We load the crabs data and transform the output into {-1,+1}
[Xtrain,ytrain,Xtest,ytest] = makeCrabs;
ytrainM1P1 = ytrain; ytestM1P1 = ytest;
ytrainM1P1(ytrainM1P1 == 2) = -1; ytestM1P1(ytestM1P1 == 2) = -1;
% ytrainM1P1 is ytrain but transformed into {-1,+1}

%% Cross Validate Sigma
% To begin, we will use cross validation to determine the RBF bandwidth
% parameter, sigma for the SVM. 

if(runFull) % takes 30 minutes
testFunction = @(Xtrain,ytrain,Xtest,sigma)svm_Classify(Xtrain,ytrain,Xtest,sigma);
lossFunction = 'ZeroOne';
sigmaVals = 0.5:0.5:10;
modelSelection = crossValidation('testFunction',testFunction,...
                                 'lossFunction',lossFunction,...
                                 'CVvalues'    ,sigmaVals,...
                                 'Xdata'       ,Xtrain,...
                                 'Ydata'       ,ytrainM1P1);
                                
                               
sigma = modelSelection.bestValue; 
else
    sigma = 1.5;  % cv chosen value
end
%% Assess SVM Performance
% 
[yhat,nsvecs] = svm_Classify(Xtrain,ytrainM1P1,Xtest,sigma);
nerrors = sum(yhat ~= ytestM1P1);
%%
results.crabs.SVM.nerrors = nerrors;
results.crabs.SVM.nzeroW = nsvecs;
results.crabs.SVM.sigma  = sigma;
%% Cross Validate L2 Lambda
% 
results.crabs.L2.sigma = sigma;
if(runFull)
T = chainTransformer({standardizeTransformer(false),kernelTransformer('rbf',sigma)});
m = logregDist('nclasses',2,'transformer',T);
testFunction = @(Xtrain,ytrain,Xtest,lambda)...
    mode(predict(fit(m,'X',Xtrain,'y',ytrain,'lambda',lambda,'prior','l2'),Xtest));

lossFunction = 'ZeroOne';

lambdaVals = logspace(-5,1,50);
modelSelection = crossValidation('testFunction' ,testFunction,...
                                  'lossFunction',lossFunction,...
                                  'CVvalues'    ,lambdaVals,...
                                  'Xdata'       ,Xtrain,...
                                  'Ydata'       ,ytrain);
                                 
                              
set(gca,'Xscale','log')                              
lambda = modelSelection.bestValue;
else
   lambda = 0.00039069;   % cv chosen value  
end
results.crabs.L2.lambda = lambda;
%% Assess L2 Performance
% 
T = chainTransformer({standardizeTransformer(false),kernelTransformer('rbf',sigma)});
m = logregDist('nclasses',2,'transformer',T);
m = fit(m,'X',Xtrain,'y',ytrain,'lambda',lambda,'prior','l2');
yhat = mode(predict(m,Xtest));
nerrors = sum(yhat' ~= ytest);

results.crabs.L2.nerrors = nerrors;
results.crabs.L2.nzeroW  = nnz(m.w);
%% Cross Validate L1 Lambda
% 
results.crabs.L1.sigma = sigma;
if(runFull)
T = chainTransformer({standardizeTransformer(false),kernelTransformer('rbf',sigma)});
m = logregDist('nclasses',2,'transformer',T);
testFunction = @(Xtrain,ytrain,Xtest,lambda)...
    mode(predict(fit(m,'X',Xtrain,'y',ytrain,'lambda',lambda,'prior','l1'),Xtest));

lossFunction = 'ZeroOne';

lambdaVals = 0.1:0.1:1;
modelSelection = crossValidation('testFunction' ,testFunction,...
                                  'lossFunction',lossFunction,...
                                  'CVvalues'    ,lambdaVals,...
                                  'Xdata'       ,Xtrain,...
                                  'Ydata'       ,ytrain,...
                                  'nfolds'      ,5);
                              
set(gca,'Xscale','log')                              
lambda = modelSelection.bestValue;
else
    lambda = 0.00026367; % chosen value (takes a long time to run)
end
results.crabs.L1.lambda = lambda;
%% Assess L1 Performance
% 
T = chainTransformer({standardizeTransformer(false),kernelTransformer('rbf',sigma)});
m = logregDist('nclasses',2,'transformer',T);
m = fit(m,'X',Xtrain,'y',ytrain,'lambda',lambda,'prior','l1');
yhat = mode(predict(m,Xtest));
nerrors = sum(yhat' ~= ytest);

results.crabs.L1.nerrors = nerrors;
results.crabs.L1.nzeroW  = nnz(m.w);




