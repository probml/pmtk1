classdef LinregL1L2 < Linreg
%% Elastic net Regression  (Single Variate Output)


    properties
        lambda1;
        lambda2;
        method;
    end
 
  
    %% Main methods
    methods
        function obj = LinregL1L2(varargin)
          % m = LinregL1(w, sigma2, transformer, lambda1, lambda2)
          % method is one of {'lars', 'shooting'}
            [obj.transformer, obj.w, obj.sigma2, obj.lambda1, obj.lambda2, obj.method] = ...
              processArgs(varargin,...
                        '-w'          , []                      ,... 
                        '-sigma2'     , []                      , ....
                        '-transformer', []                      ,...
                        '-lambda1'      , 0, ...
                        '-lambda2', 0, ...
                        '-method', 'lars');
        end
       
        function model = fit(model,varargin)
          % m = fit(m, X, y)
          % X(i,:) is i'th input; do *not* include a column of 1s
          % y(i) is i'th response
          [X, y] = processArgs(varargin, '-X', [], '-y', []);
          if ~isempty(model.transformer)
            [X, model.transformer] = train(model.transformer, X);
          end
          onesAdded = ~isempty(model.transformer) && addOffset(model.transformer);
         
          if(onesAdded)
            X = X(:,2:end);
          end
          lambda = [model.lambda1 model.lambda2];
          switch lower(model.method)
            case 'shooting'
              w = elasticNet(X,center(y),lambda(1),lambda(2),@LassoShooting);
            case 'lars'
              w = elasticNet(X,center(y),lambda(1),lambda(2),@larsLambda);
            otherwise
              error('%s is not a supported L1, (and hence L1L2) algorithm',algorithm);
          end
          w = w(:);
          if(onesAdded)
            w0 = mean(y)-mean(X)*(X\center(y));
            w = [w0;w];
            X = [ones(size(X,1),1),X];
          end
          model.w = w;
            
          n = size(X,1);
          if onesAdded
            X = [ones(n,1) X]; % column of 1s for w0 term
          end
          yhat = X*model.w;
          model.sigma2 = mean((yhat-y).^2);
          model.ndimsX = size(X,2);
          model.ndimsY = size(y,2);
        end

    end % methods


  
end