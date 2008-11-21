%% Logistic Regression vs Support Vector Machines


dataSets = {'crabs','fisherIris','glass','yeast'};


predictFunctionSVM = @(Xtrain,ytrain,Xtest,sigma)...
    oneVsAllClassifier('binaryClassifier',...
    @svmLightClassify,'Xtrain',Xtrain,'ytrain',ytrain,'Xtest',Xtest,'options',{sigma});

predictFunctionL1 = @(Xtrain,ytrain,Xtest,lambda,sigma)...
  mode(predict(fit(LogregDist('transformer',...
  ChainTransformer({StandardizeTransformer(false),KernelTransformer('rbf',sigma)})),...
  'X',Xtrain,'y',ytrain,'lambda',lambda,'prior','l1'),Xtest));

predictFunctionL2 = @(Xtrain,ytrain,Xtest,lambda,sigma)...
  mode(predict(fit(LogregDist('transformer',...
  ChainTransformer({StandardizeTransformer(false),KernelTransformer('rbf',sigma)})),...
  'X',Xtrain,'y',ytrain,'lambda',lambda,'prior','l2'),Xtest));












modelSpace.crabs.svm            = ModelSelection.makeModelSpace(logspace(-8,1,10));
modelSpace.crabs.l1             = ModelSelection.makeModelSpace(logspace(-8,3,10),logspace(-2,3,10));
modelSpace.crabs.l2             = ModelSelection.makeModelSpace(logspace(-4,2,30),logspace(-1,2,30));

modelSpace.fisherIris.svm       = ModelSelection.makeModelSpace(logspace(-8,1,10));
modelSpace.fisherIris.l1        = ModelSelection.makeModelSpace(logspace(-8,3,10),logspace(-2,3,10));
modelSpace.fisherIris.l2        = ModelSelection.makeModelSpace(logspace(-8,3,10),logspace(-2,3,10));

modelSpace.glass.svm            = ModelSelection.makeModelSpace(logspace(-8,1,10));
modelSpace.glass.l1             = ModelSelection.makeModelSpace(logspace(-8,3,10),logspace(-2,3,10));
modelSpace.glass.l2             = ModelSelection.makeModelSpace(logspace(-8,3,10),logspace(-2,3,10));

modelSpace.yeast.svm            = ModelSelection.makeModelSpace(logspace(-8,1,10));
modelSpace.yeast.l1             = ModelSelection.makeModelSpace(logspace(-8,3,10),logspace(-2,3,10));
modelSpace.yeast.l2             = ModelSelection.makeModelSpace(logspace(-8,3,10),logspace(-2,3,10));


precomp = struct;
precomp.crabs.svm.sigma         =  1e-6;
precomp.crabs.svm.cvScore       =  2.6;
precomp.crabs.l1.lambda         =  1.6681005e-07;
precomp.crabs.l1.simga          =  21.544347;
precomp.crabs.l1.cvScore        =  2.6;
precomp.crabs.l2.lambda         =  1e-4;
precomp.crabs.l2.sigma          =  62.1017;
precomp.crabs.l2.cvScore        =  1.6;

precomp.fisherIris.svm.sigma    = 0.1;
precomp.fisherIris.svm.cvScore  = 1;
precomp.fisherIris.l1.lambda    = 0.0129;
precomp.fisherIris.l1.simga     = 1.6681;
precomp.fisherIris.l1.cvScore   = 0.8;
precomp.fisherIris.l2.lambda    = 2.7826e-6;
precomp.fisherIris.l2.sigma     = 21.5443;
precomp.fisherIris.l2.cvScore   = 0.6;

precomp.glass.svm.sigma         = 0.1;
precomp.glass.svm.cvScore       = 16;
precomp.glass.l1.lambda         = 0.2154;
precomp.glass.l1.simga          = 0.4642;
precomp.glass.l1.cvScore        = 12;
precomp.glass.l2.lambda         = 0.012915;
precomp.glass.l2.sigma          = 5.9948;
precomp.glass.l2.cvScore        = 11.8;

precomp.yeast.svm.sigma         = NaN;
precomp.yeast.svm.cvScore       = NaN;
precomp.yeast.l1.lambda         = NaN;
precomp.yeast.l1.simga          = NaN;
precomp.yeast.l1.cvScore        = NaN;
precomp.yeast.l2.lambda         = NaN;
precomp.yeast.l2.sigma          = NaN;
precomp.yeast.l2.cvScore        = NaN;





runFull = false;
%% Crabs
load crabs;
%% SVM
if(runFull)
ms = ModelSelection('Xdata',Xtrain,'Ydata',ytrain,'models',modelSpace.crabs.svm,'predictFunction',predictFunctionSVM);
sigma = ms.bestModel{1};
cvScore = ms.sortedResults(1).score;
else
    sigma = 1e-6;
    cvScore = 2.6;
end
[yhat,nvecs] = svmLightClassify(Xtrain,ytrain,Xtest,sigma);
nerr = sum(yhat ~= ytest)
results.crabs.svm.nerr = nerr;
results.crabs.svm.nvecs = nvecs;
results.crabs.svm.cvScore = cvScore;
%% L1
if(runFull)
ms = ModelSelection('Xdata',Xtrain,'Ydata',ytrain,'models',modelSpace.crabs.l1,'predictFunction',predictFunctionL1);
lambda = ms.bestModel{1};
sigma  = ms.bestModel{2};
cvScore = ms.sortedResults(1).score;
else  
    lambda = 1.6681005e-07;
    sigma = 21.544347;
    cvScore = 2.6;
end
m = LogregDist('nclasses',2,'transformer',ChainTransformer({StandardizeTransformer(false),KernelTransformer('rbf',sigma)}));
m = fit(m,'X',Xtrain,'y',ytrain,'lambda',lambda,'prior','l1');
yhat = mode(predict(m,Xtest));

nerr = sum(yhat ~= ytest)
results.crabs.l1.nerr = nerr;
results.crabs.l1.nvecs = nnz(m.w.point);
results.crabs.l1.cvScore = cvScore;
%% L2

if(runFull)
ms = ModelSelection('Xdata',Xtrain,'Ydata',ytrain,'models',modelSpace.crabs.l2,'predictFunction',predictFunctionL2);
lambda = ms.bestModel{1};
sigma  = ms.bestModel{2};
cvScore = ms.sortedResults(1).score;
else
    lambda = 1e-4;
    sigma = 62.1017;
    cvScore = 1.6;
end
m = LogregDist('transformer',ChainTransformer({StandardizeTransformer(false),KernelTransformer('rbf',sigma)}));
m = fit(m,'X',Xtrain,'y',ytrain,'lambda',lambda,'prior','l2');
yhat = mode(predict(m,Xtest));
nerr = sum(yhat ~= ytest)
results.crabs.l2.nerr = nerr;
results.crabs.l2.nvecs = nnz(m.w.point);
results.crabs.l2.cvScore = cvScore;
%% IRIS
load fisherIris
rand('twister',0);
perm = randperm(150);
meas = meas(perm,:);
species = species(perm);
Xtrain = meas(1:135,:);
ytrain = species(1:135);
Xtest  = meas(136:end,:);
ytest  = species(136:end);
clear meas species perm
%% SVM
if(runFull)

ms = ModelSelection('Xdata',Xtrain,'Ydata',ytrain,'models',modelSpace.fisherIris.svm,'predictFunction',predictFunctionSVM);
sigma = ms.bestModel{1};
cvScore = ms.sortedResults(1).score;
else
    sigma = 0.1;
    cvScore = 1;
end
[yhat,nvecs] = oneVsAllClassifier('binaryClassifier',@svmLightClassify,'Xtrain',Xtrain,'ytrain',ytrain,'Xtest',Xtest,'options',{sigma});
nvecs = mean(cell2mat(nvecs));
nerr = sum(canonizeLabels(yhat) ~= canonizeLabels(ytest))
results.iris.svm.nerr = nerr;
results.iris.svm.nvecs = nvecs;
results.iris.svm.cvScore = cvScore;
%% L1
if(runFull)
ms = ModelSelection('Xdata',Xtrain,'Ydata',ytrain,'models',modelSpace.fisherIris.l1,'predictFunction',predictFunctionL1);
lambda = ms.bestModel{1};
sigma  = ms.bestModel{2};
cvScore = ms.sortedResults(1).score;
else  
    lambda = 0.0129;
    sigma = 1.6681;
    cvScore = 0.8;
end
m = LogregDist('nclasses',3,'transformer',ChainTransformer({StandardizeTransformer(false),KernelTransformer('rbf',sigma)}));
m = fit(m,'X',Xtrain,'y',ytrain,'lambda',lambda,'prior','l1');
yhat = mode(predict(m,Xtest));

nerr = sum(canonizeLabels(yhat) ~= canonizeLabels(ytest))
results.iris.l1.nerr = nerr;
results.iris.l1.nvecs = nnz(m.w.point);
results.iris.l1.cvScore = cvScore;
%% L2
if(runFull)
modelSpace = ModelSelection.makeModelSpace(logspace(-8,3,10),logspace(-2,3,10));
ms = ModelSelection('Xdata',Xtrain,'Ydata',ytrain,'models',modelSpace,'predictFunction',predictFunctionL1);
lambda = ms.bestModel{1};
sigma  = ms.bestModel{2};
cvScore = ms.sortedResults(1).score;
else  
    lambda = 2.7826e-6;
    sigma = 21.5443;
    cvScore = 0.6;
end
m = LogregDist('nclasses',3,'transformer',ChainTransformer({StandardizeTransformer(false),KernelTransformer('rbf',sigma)}));
m = fit(m,'X',Xtrain,'y',ytrain,'lambda',lambda,'prior','l1');
yhat = mode(predict(m,Xtest));

nerr = sum(canonizeLabels(yhat) ~= canonizeLabels(ytest))
results.iris.l2.nerr = nerr;
results.iris.l2.nvecs = nnz(m.w.point);
results.iris.l2.cvScore = cvScore;

%% Glass
load fglass
%% SVM
if(runFull)

modelSpace = ModelSelection.makeModelSpace(logspace(-8,1,10));
ms = ModelSelection('Xdata',Xtrain,'Ydata',ytrain,'models',modelSpace,'predictFunction',predictFunctionSVM);
sigma = ms.bestModel{1};
cvScore = ms.sortedResults(1).score;
else
    sigma = 0.1;
    cvScore = 16;
end
[yhat,nvecs] = oneVsAllClassifier('binaryClassifier',@svmLightClassify,'Xtrain',Xtrain,'ytrain',ytrain,'Xtest',Xtest,'options',{sigma});
nvecs = mean(cell2mat(nvecs));
nerr = sum(canonizeLabels(yhat) ~= canonizeLabels(ytest))
results.glass.svm.nerr = nerr;
results.glass.svm.nvecs = nvecs;
results.glass.svm.cvScore = cvScore;
%% L1
if(runFull)
modelSpace = ModelSelection.makeModelSpace(logspace(-8,3,10),logspace(-2,3,10));
ms = ModelSelection('Xdata',Xtrain,'Ydata',ytrain,'models',modelSpace,'predictFunction',predictFunctionL1);
lambda = ms.bestModel{1};
sigma  = ms.bestModel{2};
cvScore = ms.sortedResults(1).score;
else  
    lambda = 0.2154;
    sigma = 0.4642;
    cvScore = 12;
end
m = LogregDist('nclasses',6,'transformer',ChainTransformer({StandardizeTransformer(false),KernelTransformer('rbf',sigma)}));
m = fit(m,'X',Xtrain,'y',ytrain,'lambda',lambda,'prior','l1');
yhat = mode(predict(m,Xtest));

nerr = sum(canonizeLabels(yhat) ~= canonizeLabels(ytest))
results.glass.l1.nerr = nerr;
results.glass.l1.nvecs = nnz(m.w.point);
results.glass.l1.cvScore = cvScore;


%% L2
if(runFull)
modelSpace = ModelSelection.makeModelSpace(logspace(-8,3,10),logspace(-2,3,10));
ms = ModelSelection('Xdata',Xtrain,'Ydata',ytrain,'models',modelSpace,'predictFunction',predictFunctionL2);
lambda = ms.bestModel{1};
sigma  = ms.bestModel{2};
cvScore = ms.sortedResults(1).score;
else  
    lambda = 0.012915;
    sigma = 5.9948;
    cvScore = 11.8;
end
m = LogregDist('nclasses',6,'transformer',ChainTransformer({StandardizeTransformer(false),KernelTransformer('rbf',sigma)}));
m = fit(m,'X',Xtrain,'y',ytrain,'lambda',lambda,'prior','l2');
yhat = mode(predict(m,Xtest));

nerr = sum(canonizeLabels(yhat) ~= canonizeLabels(ytest))
results.glass.l2.nerr = nerr;
results.glass.l2.nvecs = nnz(m.w.point);
results.glass.l2.cvScore = cvScore;
%% Yeast
load yeastUCI
perm = randperm(1484);
X = X(perm,:);
y = y(perm,:);
Xtrain = X(1:742,:);
ytrain = y(1:742,:);
Xtest  = X(743:end,:);
ytest  = y(743:end,:);
clear X y source perm
%% SVM
if(runFull)
modelSpace = ModelSelection.makeModelSpace(logspace(-8,1,10));
ms = ModelSelection('Xdata',Xtrain,'Ydata',ytrain,'models',modelSpace,'predictFunction',predictFunctionSVM);
sigma = ms.bestModel{1};
cvScore = ms.sortedResults(1).score;
else
    sigma = NaN;
    cvScore = NaN;
end
[yhat,nvecs] = oneVsAllClassifier('binaryClassifier',@svmLightClassify,'Xtrain',Xtrain,'ytrain',ytrain,'Xtest',Xtest,'options',{sigma});
nvecs = mean(cell2mat(nvecs));
nerr = sum(canonizeLabels(yhat) ~= canonizeLabels(ytest))
results.yeast.svm.nerr = nerr;
results.yeast.svm.nvecs = nvecs;
results.yeast.svm.cvScore = cvScore;
%% L1
if(runFull)
modelSpace = ModelSelection.makeModelSpace(logspace(-8,3,10),logspace(-2,3,10));
ms = ModelSelection('Xdata',Xtrain,'Ydata',ytrain,'models',modelSpace,'predictFunction',predictFunctionL1);
lambda = ms.bestModel{1};
sigma  = ms.bestModel{2};
cvScore = ms.sortedResults(1).score;
else  
    lambda = NaN;
    sigma = NaN;
    cvScore = NaN;
end
m = LogregDist('nclasses',10,'transformer',ChainTransformer({StandardizeTransformer(false),KernelTransformer('rbf',sigma)}));
m = fit(m,'X',Xtrain,'y',ytrain,'lambda',lambda,'prior','l1');
yhat = mode(predict(m,Xtest));

nerr = sum(canonizeLabels(yhat) ~= canonizeLabels(ytest))
results.yeast.l1.nerr = nerr;
results.yeast.l1.nvecs = nnz(m.w.point);
results.yeast.l1.cvScore = cvScore;


%% L2
if(runFull)
modelSpace = ModelSelection.makeModelSpace(logspace(-8,3,10),logspace(-2,3,10));
ms = ModelSelection('Xdata',Xtrain,'Ydata',ytrain,'models',modelSpace,'predictFunction',predictFunctionL2);
lambda = ms.bestModel{1};
sigma  = ms.bestModel{2};
cvScore = ms.sortedResults(1).score;
else  
    lambda = NaN;
    sigma = NaN;
    cvScore = NaN;
end
m = LogregDist('nclasses',10,'transformer',ChainTransformer({StandardizeTransformer(false),KernelTransformer('rbf',sigma)}));
m = fit(m,'X',Xtrain,'y',ytrain,'lambda',lambda,'prior','l2');
yhat = mode(predict(m,Xtest));

nerr = sum(canonizeLabels(yhat) ~= canonizeLabels(ytest))
results.yeast.l2.nerr = nerr;
results.yeast.l2.nvecs = nnz(m.w.point);
results.yeast.l2.cvScore = cvScore;










