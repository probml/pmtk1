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
        %d = ndimensions(ML.models{1});
        d = length(ML.models{1}.w);
        W = zeros(d,Nm); w0 = zeros(1,Nm);
        sigma2 = zeros(1,Nm); nnz = zeros(1,Nm);
        for m=1:Nm
          W(:,m) = ML.models{m}.w(:);
          w0(m) = ML.models{m}.w0;
          sigma2(m) = ML.models{m}.sigma2;
          nnz(m) = sum(abs(ML.models{m}.w) ~= 0);
        end
       end
      
    end
    
    methods(Access = 'protected')
      
      function [models] = fitManyModels(ML, D)
        X = D.X; y = D.Y; 
        [n,d] = size(X);
        ndx = ind2subv(2*ones(1,d),1:(2^d))-1; % bit vectors
        m = 1;
         [XC, xbar] = center(X);
         [yC, ybar] = center(y);
         lambda = 0.001; % for numerical stability
        for i=1:size(ndx,1)
          include  = find(ndx(i,:));
          if length(include) > ML.maxSize
            continue
          end
          if ML.verbose
            modelStr = sprintf('%d ', include);
            fprintf('fitting model %d (%s)\n', m, modelStr);
          end
          
          if 1
            % for speed, we copy the fitting code from LinregL2 here
            di = length(include);
            if di==0
              w = [];
              w0 = ybar;
            else
              XX  = [XC(:,include); sqrt(lambda)*eye(di)];
              yy = [yC; zeros(di,1)];
              w  = XX \ yy; % QR
              w0 = ybar - xbar(include)*w;
            end           
            ww = [w(:); w0];
            X1 = [X(:,include) ones(n,1)]; % column of 1s for w0 term
            yhat = X1*ww;
            sigma2 = mean((yhat-y).^2);
            models{m} = LinregL2('-w', w, '-w0', w0, '-sigma2', sigma2, '-lambda',lambda);
          end
          
          if 0 % debug
            % create object then fit it - slow
            Dtmp = DataTable(X(:,include),y);
            tmp = LinregL2('-lambda', 0.001);
            tmp = fit(tmp, Dtmp);
            assert(approxeq(tmp.w, models{m}.w))
            assert(approxeq(tmp.w0, models{m}.w0))
            assert(approxeq(tmp.sigma2, models{m}.sigma2))
            %models{m} = tmp;
          end
          
          % zero-pad
          w = models{m}.w;
          wpad = zeros(d,1); wpad(include) = w;
          models{m}.w = wpad;
          m = m + 1;
        end
      end % fitManyModels

    end % methods

    methods(Static = true)   
    end % Static
end % class
