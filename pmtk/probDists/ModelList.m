classdef ModelList 
    % List of different models. We fit them all.
    % Subsequent calls to logprob/ predict/ impute use either best plugin model
    % or use Bayesian model averaging.
    
    properties
      models;
      bestModel; % plugin
      selMethod = 'bic';
      predMethod = 'plugin'; 
      nfolds = 5; LLmean; LLse; % for CV
      loglik; penloglik; posterior; % for BIC etc
      occamWindowThreshold = 0;
      verbose = false;
    end
    
    %%  Main methods
    methods
        function obj = ModelList(varargin)
          % ModelList(models, selMethod, nfolds, predMethod, occamWindowThreshold)
          % models is a cell array
          % selMethod - 'bic' or 'aic' or 'loglik' or 'cv' [default cv]
          % nfolds - number of folds [default 5]
          % predMethod - 'plugin' or 'bma'
          % occamWindowThreshold - for bma, use all models within this pc
          % of best model; 0 means use all models, 0.9 means use top 10%
          if nargin == 0; return; end
          [obj.models, obj.selMethod, obj.nfolds, ...
            obj.predMethod, obj.occamWindowThreshold] = processArgs(varargin, ...
            '-models', {}, '-selMethod', '-cv', '-nfolds', 5, ...
            '-predMethod', 'plugin', '-occamWindowThreshold', 0);
        end
        
        function mlist = fit(mlist, varargin)
          % m = fit(m, X) or fit(m, X, y)
          % Stores best model in m.bestModel.
          % For BIC, updates m.models with fitted params.
          % Also stores vector of loglik/ penloglik (for BIC etc)
          % or LLmean/ LLse (for CV)
          [X, y] = processArgs(varargin, '-X', [], '-y', []);
          Nx = size(X,1);
          switch lower(mlist.selMethod)
            case 'cv', [mlist.models, mlist.bestModel, mlist.LLmean, mlist.LLse] = ...
                selectCV(mlist, X, y);
            otherwise
              switch lower(mlist.selMethod)
                case 'bic', pen = log(Nx)/2;
                case 'aic',  pen =  Nx/2;
                case 'loglik', pen = 0; % for log marginal likelihood
              end
              [mlist.models, mlist.bestModel, mlist.loglik, mlist.penloglik] = ...
                selectPenLoglik(mlist, X, y, pen);
              mlist.posterior = exp(normalizeLogspace(mlist.penloglik));
          end 
        end
                    
        function ll = logprob(mlist, varargin)
          % ll(i) = logprob(m, X) or logprob(m, X, y)
          [X, y] = processArgs(varargin, '-X', [], '-y', []);
          nX = size(X,1);
          if isempty(y)
            fun = @(m) logprob(m, X);
          else
             fun = @(m) logprob(m, X, y);
          end
          switch mlist.predMethod
            case 'plugin'
              ll = fun(mlist.bestModel);
            case 'bma'
              maxPost = max(mlist.posterior);
              f = mlist.occamWindowThreshold;
              ndx = find(mlist.posterior >= f*maxPost);
              nM = length(ndx);
              loglik  = zeros(nX, nM);
              logprior = log(mlist.posterior(ndx));
              logprior = repmat(logprior, nX, 1);
              for m=1:nM
                loglik(:,m) = fun(mlist.models{ndx(m)});
              end
              ll = logsumexp(loglik + logprior, 2);
          end % switch
        end % funciton
        
       function [models] = fitManyModels(ML, X, y)
        % May be overriden in subclass if efficient method exists
        % for computing full regularization path
        models = ML.models;
        Nm = length(models);
        for m=1:Nm
          if isempty(y)
            models{m} = fit(models{m},  X);
          else
            models{m} = fit(models{m}, X, y);
          end
        end
       end % fitManyModels
        
       function [models, bestModel, loglik, penLL] = selectPenLoglik(ML, X, y, penalty)
        models = fitManyModels(ML, X, y);
        Nm = length(models);
        penLL = zeros(1, Nm);
        loglik = zeros(1, Nm);
        for m=1:Nm % for every model
          if isempty(y)
            loglik(m) = sum(logprob(models{m}, X),1);
          else
            loglik(m) = sum(logprob(models{m}, X, y),1);
          end
          penLL(m) = loglik(m) - penalty*dof(models{m}); %#ok 
        end
        bestNdx = argmax(penLL);
        bestModel = models{bestNdx};
       end % selectPenLoglik
      
       function [models, bestModel, LLmean, LLse] = selectCV(ML, X, y)
         Nfolds = ML.nfolds;
         Nx = size(X,1);
         randomizeOrder = true;
         [trainfolds, testfolds] = Kfold(Nx, Nfolds, randomizeOrder);
         LL = [];
         complexity = [];      
         for f=1:Nfolds % for every fold
           if ML.verbose, fprintf('starting fold %d of %d\n', f, Nfolds); end
           Xtrain = X(trainfolds{f},:); Xtest = X(testfolds{f},:);
           ytrain = y(trainfolds{f},:); ytest = y(testfolds{f},:);
           models = fitManyModels(ML, Xtrain, ytrain);
           Nm = length(models);
           for m=1:Nm
             complexity(m) = dof(models{m}); %#ok
             if isempty(y)
               ll = logprob(models{m}, Xtest);
             else
               ll = logprob(models{m}, Xtest, ytest);
             end
             LL(testfolds{f},m) = ll; %#ok
           end
         end % f
         LLmean = mean(LL,1);
         LLse = std(LL,0,1)/Nx;
         bestNdx = oneStdErrorRule(-LLmean, LLse, complexity);
         %bestNdx = argmax(LLmean);
         % Now refit all models to all the data.
         % Typically we just refit the chosen model
         % but we may want compare all models.
         % The extra cost is negligible since we've already fit
         % all models many times...
         ML.models = models;
         models = fitManyModels(ML, X, y);
         bestModel = models{bestNdx};
       end


    end % methods 
    
    methods(Static = true)  
      
      
      
    end

end