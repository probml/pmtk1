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
global_options.MaxFunEvals = 250;
%global_options.TolFun = 1e-16;
%global_options.TolX = 1e-16;
global_options.Display = 'Full';

if 1 % compare all methods
    % Steepest Descent
    solver{i} = @minFunc;
    options{i} = global_options;
    options{i}.Method = 'sd';
    i = i + 1;

    % Conjugate Gradient
    solver{i} = @minFunc;
    options{i} = global_options;
    options{i}.Method = 'cg';
    i = i + 1;

    % Barzilai & Borwein
    solver{i} = @minFunc;
    options{i} = global_options;
    options{i}.Method = 'bb';
    i = i + 1;

    % Hessian-Free Newton
    solver{i} = @minFunc;
    options{i} = global_options;
    options{i}.Method = 'pnewton0';
    i = i + 1;

    % L-BFGS
    solver{i} = @minFunc;
    options{i} = global_options;
    options{i}.Method = 'lbfgs';
    i = i + 1;

    % BFGS
    solver{i} = @minFunc;
    options{i} = global_options;
    options{i}.Method = 'bfgs';
    i = i + 1;

    % Newton
    solver{i} = @minFunc;
    options{i} = global_options;
    options{i}.Method = 'newton';
    i = i + 1;
    
        % Tensor
    solver{i} = @minFunc;
    options{i} = global_options;
    options{i}.Method = 'tensor';
    i = i + 1;
    
elseif 1 % Test Output Function
    
    solver{i} = @minFunc;
    options{i} = global_options;
    i = i + 1;

        % BFGS
%     solver{i} = @minFunc;
%     options{i} = global_options;
%     options{i}.outputFcn = @outputFcn_example;
%     i = i + 1;

elseif 0 % CG Methods
    
    solver{i} = @minFunc;
    options{i} = global_options;
    options{i}.Method = 'cg';
    options{i}.cgUpdate = 1;
    i = i + 1;
    
    solver{i} = @minFunc;
    options{i} = global_options;
    options{i}.Method = 'cg';
    options{i}.cgUpdate = 1;
    options{i}.useComplex = 1;
    i = i + 1;

else % Test Line Search initialization
        solver{i} = @minFunc;
    options{i} = global_options;
    options{i}.method = 'newton';
    i = i + 1;
    
         solver{i} = @minFunc;
    options{i} = global_options;
    options{i}.method = 'newton';
    options{i}.HessianIter = 2;
    i = i + 1;
    
         solver{i} = @minFunc;
    options{i} = global_options;
    options{i}.method = 'newton';
    options{i}.HessianIter = 5;
    i = i + 1;
    
         solver{i} = @minFunc;
    options{i} = global_options;
    options{i}.method = 'newton';
    options{i}.HessianIter = 10;
    i = i + 1;

end

for d = 1:length(dataSets)
    % Load data set
    dataSets{d}
    X=loadd(dataSets{d});
    y = full(X(:,end));
    y(y==2) = -1;
    y(y==0) = -1;
    X = full(X(:,1:end-1));
    X = standardizeCols(X);
    X = [ones(size(X,1),1) X];
    [n,p] = size(X);

    % Load params
    for o = 1:length(options)
        optim = solver{o};
        [w fval(1,o) exitflag output] = optim(@LogisticLoss,zeros(p,1),options{o},X,y);
        evals(1,o) = output.funcCount;
%        pause;
    end

    fval
    evals
    pause;
    
%        compactTest(@LLoss,zeros(p,1),options{o},X,y)
%    pause;
end