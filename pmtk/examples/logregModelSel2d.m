%% Find lambda and sigma params for kernel logistic regression
% These values are then used in logregVisualizePredictive
function [lambdaL2, sigmaL2, lambdaL1, sigmaL1] = logregModelSel2d()
%% Load the Data
% Load synthetic data generated from a mixture of Gaussians. Source:
% <<http://research.microsoft.com/~cmbishop/PRML/webdatasets/datasets.htm>>

load bishop2class
D = DataTable(X,Y);

%lambdaRange = logspace(-2,1.5,30);
%sigmaRange = 0.1:0.1:2;
%lambdaL2 = 7.8805;
%sigmaL2 = 0.2;
lambdaRange = logspace(-1,1,5);
sigmaRange = 0.1:0.1:0.3;
[lambdaL2, sigmaL2] = helper(lambdaRange, sigmaRange, D, 'L2');

%lambdaRange = logspace(-1,1,10);
%sigmaRange = [0.2,0.5:0.5:4];
lambdaRange = [0.5, 1, 2, 5]; 
sigmaRange = [0.1, 0.2,0.4];
% lambdaL1 = 2.1544;
%    sigmaL1 = 0.2;
[lambdaL1, sigmaL1] = helper(lambdaRange, sigmaRange, D, 'L1');


end

function [lambda, sigma] = helper(lambdaRange, sigmaRange, D, reg)

nl = length(lambdaRange);
ns  = length(sigmaRange);
nm = nl*ns;
models = cell(1, nm);
m = 1;
for li = 1:nl
  for si = 1:ns
    lambda = lambdaRange(li);
    sigma = sigmaRange(si);
    T = ChainTransformer({StandardizeTransformer(false), KernelTransformer('rbf', sigma)});
    switch reg
      case 'L2',
        models{m} = LogregL2('-lambda', lambda, '-transformer', T, '-addOffset', false);
      case 'L1',
        models{m} = LogregL1('-lambda', lambda, '-transformer', T, '-addOffset', false);
    end
    sigmaModel(m) = sigma; %#ok
    lambdaModel(m) = lambda; %#ok
    m = m + 1;
  end
end

ML = ModelList('-models', models, '-selMethod', 'cv', '-nfolds', 2, '-verbose', true);
ML = fit(ML, D);
lambda = ML.bestModel.lambda;
%sigma = ML.bestModel.transformer.sigma;  ChainTransformer has no sigma field
sigma = sigmaModel(ML.bestNdx);
end
