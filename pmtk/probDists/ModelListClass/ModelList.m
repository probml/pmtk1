classdef ModelList 
    % List of different models. Fitting means picking best model.
    % Subsequent calls to logprob/ predict use best model.
    
    properties
      models;
      selectionMethod;
      nfolds;
      bestModel;
      %scores; % want to maximize this
      LLmean; LLse; % for CV
      loglik; penloglik; % for BIC etc
    end
    
    %%  Main methods
    methods
        function obj = ModelList(varargin)
          % ModelList(models, selMethod, nfolds)
          % models is a cell array
          % selMethod - 'bic' or 'aic' or 'loglik' or 'cv' [default cv]
          % nfolds - number of folds [default 5]
          if nargin == 0; return; end
          [obj.models, obj.selectionMethod, obj.nfolds] = processArgs(varargin, ...
            '-models', {}, '-selMethod', '-cv', '-nfolds', 5);
        end
        
        function mlist = fit(mlist, varargin)
          % m = fit(m, X) or fit(m, X, y)
          % Stores best model in m.bestModel.
          % For BIC, updates m.models with fitted params.
          % Also stores vector of loglik/ penloglik (for BIC etc)
          % or LLmean/ LLse (for CV)
          [X, y] = processArgs(varargin, '-X', [], '-y', []);
          Nx = size(X,1);
          switch lower(mlist.selectionMethod)
            case 'cv', [mlist.bestModel, mlist.LLmean, mlist.LLse] = ...
                selectCV(mlist.models, X, y, mlist.nfolds);
            case 'bic',  [mlist.models, mlist.bestModel, mlist.loglik, mlist.penloglik] = ...
                selectPenLoglik(mlist.models, X, y, log(Nx)/2);
            case 'aic',  [mlist.models, mlist.bestModel, mlist.loglik, mlist.penloglik] = ...
                selectPenLoglik(mlist.models, X, y, Nx/2);
            case 'loglik',  [mlist.models, mlist.bestModel, mlist.loglik, mlist.penloglik] = ...
                selectPenLoglik(mlist.models, X, y, 0);
            otherwise
              error(['unknown method ' mlist.selectionMethod]);
          end 
        end
                    
        function ll = logprob(mlist, varargin)
          % ll(i) = logprob(m, X) or logprob(m, X, y)
          [X, y] = processArgs(varargin, '-X', [], '-y', []);
          if isempty(y)
           ll = logprob(mlist.bestModel, X);
          else
            ll = logprob(mlist.bestModel, X, y);
          end
        end
        
        function py = predict(mlist, varargin)
          % py = predict(m, X)
          [X] = processArgs(varargin, '-X', []);
          py = predict(mlist.bestModel, X);
        end
        
    end % methods 
    
 
end