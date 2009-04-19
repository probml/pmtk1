classdef ModelList 
    % list of different models
    
    properties
      models;
      selectionMethod;
      nfolds;
      bestModel;
      scores;
    end
    
    %%  Main methods
    methods
        function obj = ModelList(varargin)
          % ModelList(models, selMethod, nfolds)
          % models is a cell array
          % selMethod - 'bic' or 'cv' [default cv]
          % nfolds - number of folds [default 5]
          [obj.models, obj.selectionMethod, obj.nfolds] = processArgs(varargin, ...
            'models', {}, 'selMethod', 'cv', 'nfolds', 5);
        end
        
        function [bestKID score] = cvMixFactored(model, X, Nfolds)
             % cross validation of mixture of trees
              Nx = size(X,1);
              setSeed(1);
              randomizeOrder = true;
              [trainfolds, testfolds] = Kfold(Nx, Nfolds, randomizeOrder);
              Nm = length(model.numMixs);
              NLL = zeros(Nfolds, Nm);
              tim = zeros(Nfolds, Nm);
              for m=1:Nm % for every mixture                  
                  for f=1:Nfolds % for every fold
                      Xtrain = X(trainfolds{f},:);
                      Xtest = X(testfolds{f},:);
                      M = fit(model.mixDists{m}, 'data', Xtrain);
                      ll = logprob(M, Xtest);
                      NLL(f,m) = -sum(ll); %#ok
                      tim(f,m) = toc; %#ok
                  end
              end
              % use Negative Loglik as selection criterion
              [bestNLL bestKID] =min(sum(NLL, 1));
              score = sum(NLL,1);
        end
        
        function [bestModel, scores] = bic(model, X)
          [N] = size(X, 1);
          Nmodels = length(model.models);
          BICs = zeros(1, Nmodels);
          for i=1:Nmodels
            model.models{i} = fit(model.models{i}, 'data',X);
            ll = sum( logprob(model.mixDists{i},X) );
            
            Nm = model.numMixs(i);
            pc = Nm*(nStates-1) + Nm-1;
            BICs(i) = ll - 0.5*log(N)*nparams(model.models{i});
           ;
          end
          disp(BICs);
          [bestscore, model.bestKID] = max(BICs);
          model.scores = BICs;
        end

                        
        function model = fit(model, varargin)
          % Find the ML estimate of the parameters of the CPTs.
          % If the structure is unknown, find the MLE structure first.
           % 'data' - X(i,j) is value of node j in case i, i=1:n, j=1:d
            [X Nfolds] = process_options(varargin, ...
                'data', [], 'Nfolds', 4);
            nStates = max(max(X)) + 1;
            if length(model.numMixs) == 1, % just fit this model
                model.bestKID = 1;  model.scores = 0;
            else  % otherwise need to do selection
                switch lower(model.selectionMethod(1))
                    case 'c' % N-fold CV
                        [model.bestKID model.scores] = cvMixFactored(model, X, Nfolds);
                    case 'b' % BIC
                        [N] = size(X, 1);
                        BICs = zeros(length(model.numMixs),1);
                        % compute BIC to mixture
                        for i=1:length(model.numMixs)
                            model.mixDists{i} = fit(model.mixDists{i}, 'data',X);
                            % need the loglik and prameter counts 
                            ll = sum( logprob(model.mixDists{i},X) );
                            Nm = model.numMixs(i);
                            pc = Nm*(nStates-1) + Nm-1;
                            BICs(i) = ll - log(N)/2*(pc);
                        end
                        disp(BICs);
                        [bestscore, model.bestKID] = max(BICs);
                        model.scores = BICs;
                end
            end
            
            % fit the model after model selection
            [model.mixDists{model.bestKID}] = ...
                fit(model.mixDists{model.bestKID}, 'data',X);
        end
        
        function ll = logprob(model, X)
            % just evaluate the ll for the best model
           ll = logprob(model.mixDists{model.bestKID}, X);
        end
        
        function plotGraph(model, varargin)
          [nodeLabels] = process_options(varargin, 'nodeLabels', 1:nnodes(model.G));
          Graphlayout('adjMatrix', full(model.G.adjMat), 'nodeLabels', nodeLabels);
        end

        
    end % methods 
end