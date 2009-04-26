classdef LinregL1 < Linreg
%% Lasso Regression  (Single Variate Output)


    properties
        lambda;
        method;
    end
 
  
    %% Main methods
    methods
      function obj = LinregL1(varargin)
        % m = LinregL1(lambda, transformer, w, sigma2, method)
        % method is one of {'lars', 'shooting'}
        [ obj.lambda, obj.transformer, obj.w, obj.sigma2,obj.method] = processArgs(varargin,...
          '-lambda'      , 0, ...
          '-transformer', []                      ,...
          '-w'          , []                      ,...
          '-sigma2'     , []                      , ....
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
         
          switch lower(model.method)
            case 'shooting'
              model.w = LassoShooting(X,y,model.lambda);
            case 'lars'
              if(onesAdded)
                w = larsLambda(X(:,2:end),center(y),model.lambda)';
                w0 = mean(y)-mean(X)*(X\center(y));
                model.w = [w0;w];
              else
                model.w = larsLambda(X,y,model.lambda)';
              end
            otherwise
              error('%s is not a supported L1 algorithm',algorithm);
          end
         
          yhat = X*model.w;
          model.sigma2 = mean((yhat-y).^2);
          model.ndimsX = size(X,2);
          model.ndimsY = size(y,2);
        end

    end % methods
  
end % class

function w = LassoShooting2(X,y,lambda)
disp('foo')
w = LassoShooting(X,y,lambda);
end

