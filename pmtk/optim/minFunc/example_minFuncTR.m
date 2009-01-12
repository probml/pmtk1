clear all

i = 1;
dataSets{i} = 'uci.yugoBreast.mat';i=i+1;
dataSets{i} = 'uciBreast.mat';i=i+1;
dataSets{i} = 'glennHousing.mat';i=i+1;
dataSets{i} =    'statlog.heart.data';i=i+1;
dataSets{i} =    'uci.pima.data';i=i+1;
dataSets{i} =    'statlog.australianCredit.data';i=i+1;
dataSets{i} = 'uci.mushroom.mat';i=i+1;
dataSets{i} =    'glennSonar.mat';i=i+1;
dataSets{i} =    'uci.ionosphere.data';i=i+1;
dataSets{i} =    'uci.german.data';i=i+1;
dataSets{i} =    'glennBright.mat';i=i+1;
dataSets{i} =    'glennDim.mat';i=i+1;
dataSets{i} =    'glennAdult.mat';i=i+1;
dataSets{i} =    'glennCensus.mat';i=i+1;
dataSets{i} =    'delve.2norm.data';i=i+1;
dataSets{i} = 'mnist23.mat';i=i+1;

i = 1;
global_options.maxIter = 250;
global_options.Display = 'full';

% options{i} = global_options;
% options{i}.solver = 'cauchy';
% options{i}.Hessian = 'bfgs';
% i = i + 1;

options{i} = global_options;
options{i}.solver = 'dogleg';
options{i}.Hessian = 'bfgs';
i = i + 1;

options{i} = global_options;
options{i}.Solver = 'schur';
options{i}.Hessian = 'bfgs';
i = i + 1;
% 
% options{i} = global_options;
% options{i}.solver = 'steihaug';
% options{i}.Hessian = 'exact';
% options{i}.cgSolve = 0;
% i = i + 1;
% 
% options{i} = global_options;
% options{i}.solver = 'steihaug';
% options{i}.Hessian = 'exact';
% options{i}.cgSolve = 1;
% i = i + 1;
% 
% options{i} = global_options;
% options{i}.solver = 'steihaug';
% options{i}.Hessian = 'exact';
% options{i}.cgSolve = 1;
% options{i}.useComplex = 1;
% i = i + 1;
% 
% options{i} = global_options;
% options{i}.solver = 'steihaug';
% options{i}.Hessian = 'exact';
% options{i}.cgSolve = 1;
% options{i}.HvFunc = @LogisticHv;
% i = i + 1;
% 
options{i} = global_options;
options{i}.solver = 'Loo';
options{i}.Hessian = 'bfgs';
i = i + 1;

for d = 1:length(dataSets)
    % Load data set
    dataSets{d}
    X=loadd(dataSets{d});
    y = full(X(:,end));
    y(y==2) = -1;
    y(y==0) = -1;
    X = full(X(:,1:end-1));
    X = standardizeCols(X);
    [n,p] = size(X);

    % Load params
    for o = 1:length(options)
        [wTR fval(1,o) exitflag output] = minFuncTR(@LLoss,zeros(p,1),options{o},X,y);
        %[w fval(1,o) exitflag output] = minFunc(@LLoss,zeros(p,1),options{o},X,y);
        %[wTR fval(1,o) exitflag output] = minFuncTR(@penalizedL2,zeros(p,1),options{o},@LLoss,1e-2,X,y);
        evals(1,o) = output.funcCount;
        %[wTR w]
        %pause;
    end

     fval
     evals
     pause;
end