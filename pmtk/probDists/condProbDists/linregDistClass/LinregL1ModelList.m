classdef LinregL1ModelList < ModelList
%% Lasso Regression  for multiple lambda values using LARS for speed

    properties
        lambdas;
        transformer;
    end
 
    %% Main methods
    methods
      function ML = LinregL1ModelList(varargin)
        % m = LinregL1ModelList(lambdas, X, y, nlambdas, transformer, selMethod, predMethod, nfolds,
        % verbose) 
        % eg m = LinregL1ModelList('-X', X, '-y', y, '-nlambdas', 5) 
        % eg m = LinregL1ModelList('-lambdas', 'all') computes full reg path 
        % See ModelList for explanation of arguments
        [ML.lambdas, X, y, nlambdas, ML.transformer, ML.selMethod, ML.predMethod, ...
          ML.nfolds, ML.verbose] = ...
          processArgs(varargin,...
          '-lambdas', 'all', ...
          '-X', [], ...
          '-y', [], ...
          '-nlambdas', 10, ...
          '-transformer', [], ...
          '-selMethod', 'bic', ...
          '-predMethod', 'plugin', ...
          '-nfolds', 5, ...
          '-verbose', false);
        if isempty(ML.lambdas)
          ML.lambdas = [0 logspace(0, log(LinregL1ModelList.maxLambda(X,y)), nlambdas-1)];
        end
      end
       
      function [W, sigma2, dof] = getParamsForAllModels(ML)
        % Extract parameters from models in list and convert to matrix
        % W(:,m) = weights for model m
        % sigma2(m)
        % dof(m) = degrees of freedom
        Nm = length(ML.models);
        d = ndimensions(ML.models{1});
        W = zeros(d,Nm); sigma2 = zeros(1,Nm); dof = zeros(1,Nm);
        for m=1:Nm
          W(:,m) = ML.models{m}.w(:);
          sigma2(m) = ML.models{m}.sigma2;
          dof(m) = ML.models{m}.df;
        end
      end
      
      function [models] = fitManyModels(ML, X, y)
        % We use lars to find the supports
        % but then fit an 'unbiased' estimate of the weights
        lambdas = ML.lambdas;
        if ~isempty(ML.transformer)
          [X, model.transformer] = train(ML.transformer, X);
        end
        onesAdded = ~isempty(ML.transformer) && addOffset(ML.transformer);
        if onesAdded
          X = X(:,2:end); % remove leading column of 1s
        end
        X = center(X);
        X = mkUnitVariance(X);
        y = center(y);
        if ischar(lambdas) && strcmp(lambdas, 'all') 
          W = lars(X, y, 'lasso'); % each row is a different weight vector
        else
          W = larsLambda(X,y,lambdas);
        end
        %w(i,:) corresponds to the solution given lambdas(i).
        supports = abs(W) ~= 0;
        Nm = size(W,1);
        for m=1:Nm
          % refit using ridge
          models{m} = LinregL2('-transformer', ML.transformer, '-w', w, ...
            '-sigma2', sigma2, '-df', df, '-lambda', lambda);
        end
    end % fitManyModels

    end % methods

    methods(Static = true)
      
      function lambda = maxLambda(X, y)
      lambda = norm(2*(X'*y),inf);
      end
        
    end % Static
end % class
