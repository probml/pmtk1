function [] = minFunc_demo4(objType)

if nargin < 1
    objType = 'Softmax';
end

switch lower(objType)
    case 'ls'
        fprintf('Testing Solvers for Least Squares\n');
        nInstances = 1000;
        nVars = 500;

        X = randn(nInstances,nVars);
        w = randn(nVars,1);
        y = X*w + randn(nInstances,1);

        funObj = @(w)GaussianLoss(w,X,y);

        wLS = X\y;
        minObj = funObj(wLS);
        methods = {'sd','csd','cg','bb','newton0','lbfgs','bfgs','newton'};
        w_init = zeros(nVars,1);
    case 'rosen'
        fprintf('Testing Solvers for Rosenbrock\n');
        nVars = 10;

        funObj = @rosenbrock;
        minObj = funObj(ones(nVars,1));

        if nVars <= 100
        methods = {'sd','csd','cg','scg','bb','newton0','pnewton0','lbfgs','bfgs','mnewton','newton','tensor'};
        else
            methods = {'sd','csd','cg','scg','bb','newton0','pnewton0','lbfgs','bfgs','mnewton','newton'};
        end
        w_init = zeros(nVars,1);
    case 'logistic'
        fprintf('Testing Solvers for Logistic Regression\n');
        nInstances = 250;
        nVars = 25;

        X = [ones(nInstances,1) randn(nInstances,nVars-1)];
        w = randn(nVars,1);
        %w(rand(nVars,1) < .5) = 0; % Make half the variables irrelevant
        y = sign(X*w + randn(nInstances,1));

        funObj = @(w)LogisticLoss(w,X,y);

        wLR = minFunc(funObj,zeros(nVars,1),struct('TolFun',1e-16,'TolX',1e-16,'Method','newton'));
        minObj = funObj(wLR);
        methods = {'sd','csd','cg','scg','bb','newton0','pnewton0','lbfgs','bfgs','mnewton','newton','tensor'};
        w_init = zeros(nVars,1);
    case 'softmax'
        fprintf('Testing Solvers for Multinomial Logistic Regression\n');
        nInstances = 250;
        nVars = 25;
        nClasses = 5;
        
        X = [ones(nInstances,1) randn(nInstances,nVars-1)];
        w = randn(nVars,nClasses);
        [junk y] = max(X*w,[],2);
        
        funObj = @(w)SoftmaxLoss2(w,X,y,nClasses);
        
        nVars = nVars*(nClasses-1);
        wSM = minFunc(funObj,zeros(nVars,1),struct('TolFun',1e-16,'TolX',1e-16,'Method','newton'));
        minObj = funObj(wSM);
        methods = {'sd','csd','cg','scg','bb','newton0','pnewton0','lbfgs','bfgs','mnewton','newton'};
        w_init = zeros(nVars,1);
    case 'neural'
        nInstances = 500;
        nVars = 10;
        nHidden = [10 10];

        X = randn(nInstances,nVars);

        nExamplePoints = 10;
        examplePoints = randn(nExamplePoints,nVars);
        exampleTarget = randn(nExamplePoints,1);
        for i = 1:nInstances
            dists = sum((repmat(X(i,:),nExamplePoints,1) -examplePoints).^2,2);
            dists = sum(abs(repmat(X(i,:),nExamplePoints,1) - examplePoints),2);
            lik = (1/sqrt(2*pi))*exp(-dists/2);
            lik = lik./sum(lik);
            y(i,1) = lik'*exampleTarget + randn/100;
        end
        
        nParams = nVars*nHidden(1);
        for h = 2:length(nHidden);
            nParams = nParams+nHidden(h-1)*nHidden(h);
        end
        nVars = nParams+nHidden(end);
        w_init = randn(nVars,1);
        funObj = @(w)MLPregressionLoss(w,X,y,nHidden);
        w = minFunc(funObj,w_init,struct('TolFun',1e-16,'TolX',1e-16));
        minObj = funObj(w);
        methods = {'sd','csd','cg','scg','bb','newton0','pnewton0','lbfgs','bfgs'};
    case 'kernel'
        nInstances = 250;
        nFeatures = 3;
        nExamplePoints = 5; % Set to 1 for linear classifier, higher for more non-linear
        nClasses = 2;
        examplePoints = 2*rand(nClasses*nExamplePoints,nFeatures)-1;
        X = 2*rand(nInstances,nFeatures)-1;
        for i = 1:nInstances
            dists = sum((repmat(X(i,:),nClasses*nExamplePoints,1) - examplePoints).^2,2);
            [minVal minInd] = min(dists);
            y(i,1) = sign(mod(minInd,nClasses)-.5);
        end
        K = kernelRBF(X,X,1);
        nVars = nInstances;
        w_init = zeros(nVars,1);
        lambda = 1e-5;
        funObj = @(w)penalizedKernelL2(w,K,@LogisticLoss,lambda,K,y);
        [w,minObj] = minFunc(funObj,zeros(nVars,1),struct('maxFunEvals',100,'TolFun',1e-16,'TolX',1e-16,'Method','newton'));

        methods = {'sd','csd','cg','scg','bb','newton0','pnewton0','lbfgs','bfgs','mnewton','newton'};
end

if exist('fminunc')==2
    methods = {methods{:},'fminunc-MS','fminunc-LS',};
end
if exist('minimize')==2
   methods = {'minimize',methods{:}}; 
end
methods
pause;

options.TolFun = 1e-10;
options.TolX = 1e-10;
options.maxFunEvals = 250;
options.Display = 'full';
options.bbType = 1;

figure(1);clf; hold on
colors = getColorsRGB;
symbols = getSymbols;

for i = 1:length(methods)
    if strcmp(methods{i},'minimize')
        fprintf('Running minimize.m\n');
        global fValues;
        fValues = zeros(0,1);
        w = minimize(w_init,@tracedFunObj,-options.maxFunEvals,funObj);
        figure(1);
        plot(1:length(fValues),log(abs(fValues-minObj)+1e-10),'color',colors(i,:),'Marker',symbols{i});
    elseif strcmp(methods{i},'fminunc-LS') || strcmp(methods{i},'fminunc-MS')
        fprintf('Running fminunc.m\n');
        global fValues;
        fValues = zeros(0,1);
        ops = optimset(@fminunc);
        ops.TolFun = options.TolFun;
        ops.TolX = options.TolX;
        ops.MaxFunEvals = options.maxFunEvals;
        ops.Display = options.Display;
        ops.GradObj = 'on';
        if strcmp(methods{i},'fminunc-LS')
            ops.LargeScale = 'on';
            ops.Hessian = 'on';
        else
            ops.LargeScale = 'off';
            ops.Hessian = 'off';
        end
        w = fminunc(@tracedFunObj,w_init,ops,funObj);
        figure(1);
        plot(1:length(fValues),log(abs(fValues-minObj)+1e-10),'color',colors(i,:),'Marker',symbols{i});
    else
        fprintf('Running %s\n',methods{i});
        options.Method = methods{i};
        options.Display = 'full';
        [w fval exitflag output] = minFunc(funObj,w_init,options);
        fprintf('Error in objective value: %e\n',abs(minObj-funObj(w)));
        fprintf('Maximum error parameter estimate: %e\n',max(abs(ones(nVars,1)-w)));
        fprintf('Number of function evaluations: %d\n',output.funcCount);

        figure(1);
        plot(output.trace.funcCount,log(abs(output.trace.fval-minObj)+1e-10),'color',colors(i,:),'Marker',symbols{i});
    end

    legend(methods{1:i});
    xlabel('Function Evaluation');
    ylabel('log( objective value - optimal )');
    xlim([1 options.maxFunEvals]);
    pause;
end

end

function [f,g,H] = tracedFunObj(w,funObj)
if nargout > 2
    [f,g,H] = funObj(w);
else
[f,g] = funObj(w);
end
global fValues;
fValues(end+1,1) = f;
end