classdef ModelList 
    % List of different models. Fitting means picking best model.
    % Subsequent calls to logprob/ predict use best model.
    
    properties
      models;
      selectionMethod;
      nfolds;
      bestModel;
      scores; % want to maximize this
    end
    
    %%  Main methods
    methods
        function obj = ModelList(varargin)
          % ModelList(models, selMethod, nfolds)
          % models is a cell array
          % selMethod - 'bic' or 'aic' or 'loglik' or 'cv' [default cv]
          % nfolds - number of folds [default 5]
          [obj.models, obj.selectionMethod, obj.nfolds] = processArgs(varargin, ...
            'models', {}, 'selMethod', 'cv', 'nfolds', 5);
        end
        
        function mlist = fit(mlist, varargin)
          % m = fit(m, X) or fit(m, X, y)
          [X, y] = processArgs(varargin, 'X', [], 'y', []);
          Nx = size(X,1);
          switch mlist.selectionMethod
            case 'cv', [mlist.bestModel, mlist.scores] = ...
                selectCV(mlist.models, X, y, mlist.nfolds);
            case 'bic',  [mlist.bestModel, mlist.scores] = ...
                selectPenLoglik(mlist.models, X, y, log(Nx)/2);
            case 'aic',  [mlist.bestModel, mlist.scores] = ...
                selectPenLoglik(mlist.models, X, y, Nx/2);
            case 'loglik',  [mlist.bestModel, mlist.scores] = ...
                selectPenLoglik(mlist.models, X, y, 0);
            otherwise
              error(['unknown method ' mlist.selectionMethod]);
          end 
        end
              
         
        function ll = logprob(mlist, varargin)
          % ll(i) = logprob(m, X) or logprob(m, X, y)
          [X, y] = processArgs(varargin, 'X', [], 'y', []);
          if isempty(y)
           ll = logprob(mlist.bestModel, X);
          else
            ll = logprob(mlist.bestModel, X, y);
          end
        end
        
        function py = predict(mlist, varargin)
          % py = predict(m, X)
          [X] = processArgs(varargin, 'X', []);
          py = predict(mlist.bestModel, X);
        end
        
    end % methods 
    
    methods(Access = 'protected')
      function [bestModel, LLmean, LLse] = selectCV(models, X, y, Nfolds)
        Nx = size(X,1);
        randomizeOrder = true;
        [trainfolds, testfolds] = Kfold(Nx, Nfolds, randomizeOrder);
        Nm = length(models);
        LL = zeros(Nx, Nm);
        complexity = zeros(1, Nm);
        for m=1:Nm % for every model
          complexity(m) = nparams(models{m});
          for f=1:Nfolds % for every fold
            Xtrain = X(trainfolds{f},:);
            Xtest = X(testfolds{f},:);
            if isempty(y)
              tmp = fit(models{m}, 'data', Xtrain);
              ll = logprob(tmp, Xtest);
            else
              ytrain = y(trainfolds{f},:);
              ytest = y(testfolds{f},:);
              tmp = fit(models{m}, 'X', Xtrain, 'y', ytrain);
              ll = logprob(tmp, Xtest, ytest);
            end
            LL(testfolds{f},m) = ll;
          end
        end
        LLmean = mean(LL,1);
        LLse = std(LL,0,1)/Nx;
        bestNdx = oneStdErrorRule(-LLmean, LLse, complexity);
        %bestNdx = argmax(LLmean);
        % Now fit chosen model to all the data
        if isempty(y)
          bestModel = fit(models{bestNdx}, X);
        else
          bestModel = fit(models{bestNdx}, X, y);
        end
      end

      function [bestModel, score] = selectPenLoglik(models, X, y, penalty)
        Nm = length(models);
        score = zeros(1, Nm);
        for m=1:Nm % for every model
          if isempty(y)
            models{m} = fit(models{m},  X);
            score(m) = logprob(models{m}, X);
          else
            models{m} = fit(models{m}, X,  y);
            score(m) = logprob(models{m}, X, y);
          end
          score(m) = score(m) - penalty*nparams(models{m});
        end
        bestNdx = argmax(score);
        bestModel = models{bestNdx};
      end

    end % protected methods

end