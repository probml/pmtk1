clear all

nInstances = 1000;
nVars = 25;

X = [ones(nInstances,1) rand(nInstances,nVars-1)];
w = randn(nVars,1);
y = sign(X*w + randn(nInstances,1));
flipPos = rand(nInstances,1) > .9;
y(flipPos) = -y(flipPos);

funObj = @(w)LogisticLoss(w,X,y);


fprintf('Running L-BFGS\n');
options = [];
%options.Method = 'bfgs';
%options.initialHessType = 0;
%options.SR1 = 0;
wLBFGS= minFunc(funObj,zeros(nVars,1),options);
pause;

fprintf('Running Conic Model\n');
w = conicModel(funObj,zeros(nVars,1));

