function [] = example_minFunc_tune(objType)

nTrials = inf;
plotIter = 5;

%% Objective Function type and problem size
nInstances = 500;
nVars = 100;
nClasses = 5;

%% Global options
gOptions = [];
gOptions.maxFunEvals = 250;
gOptions.Display = 'Full';
gOptions.TolX = 1e-10;
gOptions.TolFun = 1e-10;
gOptions.optTol = 1e-10;


%% Options for specific methods

m = 1;

% method{m} = 'Gradient Descent';
% options{m} = gOptions;
% options{m}.Method = 'sd';
% m = m + 1;
% 
% method{m} = 'Cyclic Steepest Descent';
% options{m} = gOptions;
% options{m}.Method = 'csd';
% m = m + 1;
% 
% method{m} = 'Barzilai Borwein';
% options{m} = gOptions;
% options{m}.Method = 'bb';
% m = m + 1;
%
% method{m} = 'Conjugate Gradient';
% options{m} = gOptions;
% options{m}.Method = 'cg';
% m = m + 1;
%
% method{m} = 'Preconditioned Conjugate Gradient';
% options{m} = gOptions;
% options{m}.Method = 'pcg';
% m = m + 1;
%
% method{m} = 'Hessian-Free Newton';
% options{m} = gOptions;
% options{m}.Method = 'newton0';
% m = m + 1;
%
% method{m} = 'Preconditioned Hessian-Free Newton';
% options{m} = gOptions;
% options{m}.Method = 'newton0lbfgs';
% m = m + 1;
%
% method{m} = 'Limited-Memory BFGS';
% options{m} = gOptions;
% options{m}.Method = 'lbfgs';
% m = m + 1;
%
% if nVars <= 1000
%     method{m} = 'BFGS';
%     options{m} = gOptions;
%     options{m}.Method = 'bfgs';
%     m = m + 1;
%
%     method{m} = 'Modified Newton';
%     options{m} = gOptions;
%     options{m}.Method = 'mnewton';
%     m = m + 1;
%
%     method{m} = 'Newton';
%     options{m} = gOptions;
%     options{m}.Method = 'newton';
%     m = m + 1;
% end
%
% if nVars <= 100
%     method{m} = 'Tensor';
%     options{m} = gOptions;
%     options{m}.Method = 'tensor';
%     m = m + 1;
% end

%% Test
    
        method{m} = 'Quasi-Newton';
    options{m} = gOptions;
    options{m}.Method = 'qnewton';
    options{m}.qnUpdate = 0;
    m = m + 1;
    
%         method{m} = 'Quasi-Newton';
%     options{m} = gOptions;
%     options{m}.Method = 'qnewton';
%     options{m}.qnUpdate = 1;
%     m = m + 1;
    
%             method{m} = 'Quasi-Newton';
%     options{m} = gOptions;
%     options{m}.Method = 'qnewton';
%     options{m}.qnUpdate = 2;
%     m = m + 1;
    
%                 method{m} = 'Quasi-Newton';
%     options{m} = gOptions;
%     options{m}.Method = 'qnewton';
%     options{m}.qnUpdate = 3;
%     m = m + 1;
    
                    method{m} = 'Quasi-Newton';
    options{m} = gOptions;
    options{m}.Method = 'qnewton';
    options{m}.qnUpdate = 5;
    %options{m}.initialHessType = 0;
    m = m + 1;
    
                        method{m} = 'Quasi-Newton';
    options{m} = gOptions;
    options{m}.Method = 'qnewton';
    options{m}.qnUpdate = 6;
    %options{m}.initialHessType = 0;
    m = m + 1;
    


%%


for t = 1:nTrials

    %% Generate Data
    X = [ones(nInstances,1) randn(nInstances,nVars-1)];

    if strcmpi(objType,'Gaussian');
        wt = randn(nVars,1);
        y = X*wt + randn(nInstances,1);
        funObj = @(w)GaussianLoss(w,X,y);
        w_init = zeros(nVars,1);
    elseif strcmpi(objType,'Logistic')
        wt = randn(nVars,1);
        wt(rand(nVars,1) < .5) = 0; 
        y = sign(X*wt);
        flipPos = rand(nInstances,1) < .1;
        y(flipPos) = -y(flipPos);
        funObj = @(w)LogisticLoss(w,X,y);
        w_init = zeros(nVars,1);
    elseif strcmpi(objType,'Rosen')
        if t > 1
           break;
        end
        funObj = @(w)rosenbrock(w);
        w_init = zeros(nVars,1);
    else
        wt = randn(nVars,nClasses);
        wt(rand(numel(wt),1) < .5) = 0; 
        [junk y] = max(X*wt,[],2);
        flipPos = rand(nInstances,1) < .1;
        y(flipPos) = ceil(nClasses*rand(sum(flipPos),1));
        %funObj = @(w)SoftmaxLoss2(w,X,y,nClasses);
        funObj = @(w)penalizedL2(w,@SoftmaxLoss2,.1,X,y,nClasses);
        w_init = zeros(nVars*(nClasses-1),1);
    end

    %% Run Methods

    for m = 1:length(method)
        fprintf('%s\n',method{m});
        
        if strcmp(objType,'Gaussian')
            options{m}.HvFunc = @(v,w)GaussianHv(v,w,X,y);
        elseif strcmp(objType,'Logistic')
            options{m}.HvFunc = @(v,w)LogisticHv(v,w,X,y);
        end
        
        if strcmp(method{m},'TR')
            [w fval(t,m) exitflag output] = minFuncTR(funObj,w_init,options{m});
        else
            [w fval(t,m) exitflag output] = minFunc(funObj,w_init,options{m});
        end
        funEvals(t,m) = output.funcCount;
        %pause;
    end
    fval
    funEvals

    if mod(t,plotIter)==0
        % Plot of deviation from minimum
        figure(1);
        clf;
        boxplot(fval-repmat(min(fval,[],2),[1 size(fval,2)]));
        pause;

        % Plot of #iterations
        figure(2);
        clf;
        boxplot(funEvals);
        pause;
        
        % Plot of # extra iterations beyond best
        figure(3);
        clf;
        boxplot(funEvals-repmat(min(funEvals,[],2),[1 size(funEvals,2)]));
        pause;
    end
end