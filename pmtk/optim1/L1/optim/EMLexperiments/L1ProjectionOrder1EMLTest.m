%load car;

load docData;
X = full(xtrain);

Y1 = oneOfK(ytrain,2);
clearvars -except X Y1
[n,d] = size(X);
nclasses = 2;


lambdaVec = 0.1*ones(d,nclasses-1);
lambdaVec = lambdaVec(:);
 
options.verbose = false;
options.order = -1;


 
 [w1,fevals1] = L1GeneralProjection(@multinomLogregNLLGradHessL2,zeros(d*(nclasses-1),1),lambdaVec,options,X,Y1,0,false);
 



 tic
 w2 = L1ProjectionOrder1EML(X,Y1,lambdaVec,false);
 toc
 
%w2 = compileAndRun('L1ProjectionOrder1EML',X,Y1,lambdaVec,false);
 
 
 tic
 [w2,fevals2] = compileRunAndSave('L1ProjectionOrder1EML',X,Y1,lambdaVec,false);
 toc

 
 
 assert(approxeq(w1,w2));
 
 