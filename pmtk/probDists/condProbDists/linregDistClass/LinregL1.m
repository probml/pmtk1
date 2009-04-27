classdef LinregL1 < Linreg
%% Lasso Regression  (Single Variate Output)


    properties
        lambda;
        method;
    end
 
  
    %% Main methods
    methods
      function obj = LinregL1(varargin)
        % m = LinregL1(lambda, transformer, w, w0, sigma2, method, df, addOffset)
        % method is one of {'shooting', 'l1_ls'}
        [ obj.lambda, obj.transformer, obj.w, obj.w0, obj.sigma2, obj.method, obj.df, obj.addOffset] = ...
          processArgs(varargin,...
          '-lambda'      , 0, ...
          '-transformer', [], ...                     
          '-w'          , [], ...  
           '-w0'          , [], ... 
          '-sigma2'     , [], ...                     
          '-method', 'shooting', ...
          '-df', [], ...
          '-addOffset', true);
      end
       
        function model = fit(model,varargin)
          % m = fit(m, X, y)
          % X(i,:) is i'th input; do *not* include a column of 1s
          % y(i) is i'th response
          [X, y] = processArgs(varargin, '-X', [], '-y', []);
          if ~isempty(model.transformer)
            [X, model.transformer] = train(model.transformer, X);
          end
          % We can center at training time but not at test time
          % since w0 will compensate (is this correct??)
          % (If we need to center at test time, use a transformer)
          [XC, xbar] = center(X);
          %XC = mkUnitVariance(XC);
          [yC, ybar] = center(y);
          switch lower(model.method)
            case 'shooting'
              w = LassoShooting(XC, yC, model.lambda, 'offsetAdded', false);
            case 'l1_ls'
              w = l1_ls(XC, yC, model.lambda, 1e-3, true);
            otherwise
              error('%s is not a supported L1 algorithm', model.method);
          end
          if model.addOffset
            w0 = ybar - xbar*w;
          else
            w0 = 0;
          end
          model.w = w; model.w0 = w0;
          n = size(X,1);
          X1 = [ones(n,1) X];
          ww = [w0; w];
          yhat = X1*ww;
          model.sigma2 = mean((yhat-y).^2);
          model.df = sum(abs(w) ~= 0); % num non zeros
          model.ndimsX = size(X,2);
          model.ndimsY = size(y,2);
        end

    end % methods
  
end % class
