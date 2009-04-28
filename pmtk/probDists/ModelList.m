classdef ModelList 
    % List of different models. We fit them all.
    % Subsequent calls to logprob/ predict/ impute use either best plugin model
    % or use Bayesian model averaging.
    
    properties
      models;
      bestModel; % plugin
      selMethod = 'bic';
      predMethod = 'plugin'; 
      nfolds = 5; costMean; costSe; % for CV
      loglik; penloglik; posterior; % for BIC etc
      occamWindowThreshold = 0;
      verbose = false;
      costFnForCV; % = (@(M,D) -logprob(M,D));
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
            obj.predMethod, obj.occamWindowThreshold, obj.costFnForCV] = processArgs(varargin, ...
            '-models', {}, '-selMethod', '-cv', '-nfolds', 5, ...
            '-predMethod', 'plugin', '-occamWindowThreshold', 0, ...
            '-costFnForCV', (@(M,D) -logprob(M,D)));
        end
        
        function mlist = fit(mlist, varargin)
          % m = fit(m, D)
          % D is a DataTable
          % Stores best model in m.bestModel.
          % For BIC, updates m.models with fitted params.
          % Also stores vector of loglik/ penloglik (for BIC etc)
          % or LLmean/ LLse (for CV)
          [D] = processArgs(varargin, '-D', []);
          Nx = ncases(D);
          switch lower(mlist.selMethod)
            case 'cv', [mlist.models, mlist.bestModel, mlist.costMean, mlist.costSe] = ...
                selectCV(mlist, D);
            otherwise
              switch lower(mlist.selMethod)
                case 'bic', pen = log(Nx)/2;
                case 'aic',  pen =  Nx/2;
                case 'loglik', pen = 0; % for log marginal likelihood
              end
              [mlist.models, mlist.bestModel, mlist.loglik, mlist.penloglik] = ...
                selectPenLoglik(mlist, D, pen);
              mlist.posterior = exp(normalizeLogspace(mlist.penloglik));
          end 
        end
                    
        function ll = logprob(mlist, varargin)
          % ll(i) = logprob(m, D) 
          % D is a DataTable
          [D] = processArgs(varargin, '-D', []);
          nX = ncases(D);
          switch mlist.predMethod
            case 'plugin'
              ll = logprob(mlist.bestModel, D);
            case 'bma'
              maxPost = max(mlist.posterior);
              f = mlist.occamWindowThreshold;
              ndx = find(mlist.posterior >= f*maxPost);
              nM = length(ndx);
              loglik  = zeros(nX, nM);
              logprior = log(mlist.posterior(ndx));
              logprior = repmat(logprior, nX, 1);
              for m=1:nM
                loglik(:,m) = logprob(mlist.models{ndx(m)}, D);
              end
              ll = logsumexp(loglik + logprior, 2);
          end % switch
        end % funciton
        
       function [models] = fitManyModels(ML, D)
        % May be overriden in subclass if efficient method exists
        % for computing full regularization path
        models = ML.models;
        Nm = length(models);
        for m=1:Nm
          models{m} = fit(models{m}, D);
        end
       end % fitManyModels
        
       function [models, bestModel, loglik, penLL] = selectPenLoglik(ML, D, penalty)
        models = fitManyModels(ML, D);
        Nm = length(models);
        penLL = zeros(1, Nm);
        loglik = zeros(1, Nm);
        for m=1:Nm % for every model
          loglik(m) = sum(logprob(models{m}, D),1);
          penLL(m) = loglik(m) - penalty*dof(models{m}); %#ok 
        end
        bestNdx = argmax(penLL);
        bestModel = models{bestNdx};
       end % selectPenLoglik
      
       function [models, bestModel, NLLmean, NLLse] = selectCV(ML, D)
         Nfolds = ML.nfolds;
         Nx = ncases(D);
         randomizeOrder = false;
         [trainfolds, testfolds] = Kfold(Nx, Nfolds, randomizeOrder);
         NLL = [];
         complexity = [];      
         for f=1:Nfolds % for every fold
           if ML.verbose, fprintf('starting fold %d of %d\n', f, Nfolds); end
           Dtrain = D(trainfolds{f});
           Dtest = D(testfolds{f});
           models = fitManyModels(ML, Dtrain);
           Nm = length(models);
           for m=1:Nm
             complexity(m) = dof(models{m}); %#ok
             nll = ML.costFnForCV(models{m}, Dtest); %logprob(models{m}, Dtest);
             NLL(testfolds{f},m) = nll; %#ok
           end
         end % f
         NLLmean = mean(NLL,1);
         NLLse = std(NLL,0,1)/sqrt(Nx);
         bestNdx = oneStdErrorRule(NLLmean, NLLse, complexity);
         %bestNdx = argmax(LLmean);
         % Now refit all models to all the data.
         % Typically we just refit the chosen model
         % but the extra cost is negligible since we've already fit
         % all models many times...
         ML.models = models;
         models = fitManyModels(ML, D);
         bestModel = models{bestNdx};
       end


    end % methods 
    
    methods(Static = true)  
     
      
    end

end