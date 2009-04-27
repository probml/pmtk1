classdef Linreg < CondProbDist
%% Linear Regression Conditional Distribution (Single Variate Output)


    properties
        w;                % weight vector
        df; % degrees of freedom
        sigma2;           % noise variance                          
        transformer;      % A data transformer object, e.g. KernelTransformer
    end

    %% Main methods
    methods
        function obj = LinregDist(varargin)
            [obj.transformer, obj.w, obj.sigma2] = processArgs(varargin,...
                        'transformer', []                      ,...      
                        'w'          , []                      ,... 
                        'sigma2'     , []);
        end
       
        function model = fit(model,varargin)
          % m = fit(m, X, y)
          % X(i,:) is i'th input; do *not* include a column of 1s
          % y(i) is i'th response
          [X, y] = processArgs(varargin, '-X', [], '-y', []);
          if ~isempty(model.transformer)
            [X, model.transformer] = train(model.transformer, X);
          end
          model.w = X \ y; % least squares
          yhat = X*model.w;
          model.sigma2 = mean((yhat-y).^2);
          model.ndimsX = size(X,2);
          model.ndimsY = size(y,2);
        end

        function py = predict(model,X)
          %  X(i,:) is i'th input
          % py(i) = p(y|X(i,:), params), a GaussDist
         
          if ~isempty(model.transformer)
            X = test(model.transformer, X);
          end
          n = size(X,1);
          muHat = X*model.w(:);
          sigma2Hat = model.sigma2*ones(n,1); % constant variance!
          py = GaussDist(muHat, sigma2Hat);
        end
  
        function model = mkRndParams(model, d)
         % Generate and set random d-dimensional parameters    
            model.w = randn(d,1);
            model.sigma2 = rand(1,1);
        end

        function np = dof(model)
          np = model.df; % length(model.w);
        end
          
        function d = ndimensions(model)
          d = length(model.w);
        end

        function p = logprob(model, X, y)
        % p(i) = log p(y(i) | X(i,:), model params)
            [yhat] = mean(predict(model, X));
            s2 = model.sigma2;
            p = -1/(2*s2)*(y(:)-yhat(:)).^2 - 0.5*log(2*pi*s2);
        end
        
        

    end


   

end