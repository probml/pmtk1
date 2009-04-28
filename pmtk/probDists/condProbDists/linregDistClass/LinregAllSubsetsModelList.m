classdef LinregAllSubsetsModelList < ModelList
%% Linear regression using all possible subsets

    properties
        transformer;
        addOffset = true;
        maxSize;
    end
 
    %% Main methods
    methods
      function ML = LinregAllSubsetsModelList(varargin)
        % m = LinregAllSubsetsModelList(maxSize, transformer, selMethod, predMethod, nfolds,
        % verbose, costFnForCV) 
        [ML.maxSize,  ML.transformer, ML.selMethod, ML.predMethod, ...
          ML.nfolds, ML.verbose, ML.costFnForCV] = ...
          processArgs(varargin,...
          '-maxSize', inf, ...
          '-transformer', [], ...
          '-selMethod', 'bic', ...
          '-predMethod', 'plugin', ...
          '-nfolds', 5, ...
          '-verbose', false, ...
           '-costFnForCV', (@(M,D) -logprob(M,D)) ...
           );
      end
       
       function [W, w0, sigma2, nnz] = getParamsForAllModels(ML)
        % Extract parameters from models in list and convert to matrix
        % W(:,m) = weights for model m
        % w0(m)
        % sigma2(m)
        % nnz(m) is num non zeros
        Nm = length(ML.models);
        d = ndimensions(ML.models{1});
        W = zeros(d,Nm); w0 = zeros(1,Nm);
        sigma2 = zeros(1,Nm); nnz = zeros(1,Nm);
        for m=1:Nm
          W(:,m) = ML.models{m}.w(:);
          w0(m) = ML.models{m}.w0;
          sigma2(m) = ML.models{m}.sigma2;
          nnz(m) = sum(abs(ML.models{m}.w) ~= 0);
        end
       end
      
      
      function [models] = fitManyModels(ML, D)
        X = D.X; y = D.Y; clear D;
        d = size(X,2);
        ndx = ind2subv(2*ones(1,d),1:(2^d))-1; % bit vectors
        m = 1;
        for i=1:size(ndx,1)
          include  = find(ndx(i,:));
          if length(include) > ML.maxSize
            continue
          end
          if ML.verbose
            modelStr = sprintf('%d ', include);
            fprintf('fitting model %d (%s)\n', m, modelStr);
          end
          Dtmp = DataTable(X(:,include),y);
          models{m} = fit(LinregL2('-lambda', 0.001), Dtmp);
          ww = models{m}.w;
          %ww = X(:,include) \ y;
          w = zeros(d,1); w(include) = ww;
          models{m}.w = w;
          m = m + 1;
        end
      end % fitManyModels

    end % methods

    methods(Static = true)   
    end % Static
end % class
