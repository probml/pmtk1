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
%dataSets{i} =    'glennAdult.mat';i=i+1;
dataSets{i} =    'glennCensus.mat';i=i+1;
dataSets{i} =    'delve.2norm.data';i=i+1;
dataSets{i} = 'mnist23.mat';i=i+1;

i = 1;
global_options.MaxIter = 2000;
global_options.MaxFunEvals = 2000;
global_options.Display = 'full';

% solver{i} = @minFunc;
% options{i} = global_options;
% i=i+1;

% solver{i} = @fminsearch;
% options{i} = global_options;
% options{i}.Display = 'iter';
% i = i + 1;

% solver{i} = @minFuncDFO;
% options{i} = global_options;
% options{i}.solver = 'random';
% i=i+1;

% solver{i} = @minFuncDFO;
% options{i} = global_options;
% options{i}.solver = 'interpModel';
% i=i+1;

% solver{i} = @minFuncDFO;
% options{i} = global_options;
% options{i}.solver = 'conjugateDirection';
% i=i+1;

% solver{i} = @minFuncDFO;
% options{i} = global_options;
% options{i}.solver = 'hookeJeeves';
% i=i+1;

solver{i} = @minFuncDFO;
options{i} = global_options;
options{i}.solver = 'coordinateSearch';
i=i+1;

solver{i} = @minFuncDFO;
options{i} = global_options;
options{i}.solver = 'coordinateSearch';
options{i}.bracket = 3;
i=i+1;

% solver{i} = @minFuncDFO;
% options{i} = global_options;
% options{i}.solver = 'patternSearch';
% i=i+1;

% solver{i} = @minFuncDFO;
% options{i} = global_options;
% options{i}.solver = 'nelderMead';
% i=i+1;

% solver{i} = @minFuncDFO;
% options{i} = global_options;
% options{i}.solver = 'numDiff';
% i=i+1;

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
        rand('state',0);
        randn('state',0);
        minimizer = solver{o};
        
        % Maximum Likelihood
        [wML fval(d,o) exitflag output] = minimizer(@LogisticLoss,zeros(p,1),options{o},X,y);
        
        % L1-Regularization
        %[wL1 fval(d,o) exitflag output] = minimizer(@LogisticLossL1,zeros(p,1),options{o},X,y,p);
        
        evals(d,o) = output.funcCount;
        %[wTR w]
        %pause;
    end
    
    % Alternative Methods
%     f = @(x) -LLoss(x,X,y);
%      x0 = zeros(p,1);
%      stopit = [1e-9 2000 inf];
%     [x,f,nf] = mdsmax(f,x0,stopit);
%     evals(d,o+1) = nf;
%     fval(d,o+1) = -f;

     fval(d,:)
     evals(d,:)
     pause;
end