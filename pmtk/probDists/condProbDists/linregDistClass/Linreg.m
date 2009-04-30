classdef Linreg < CondProbDist
%% Linear Regression Conditional Distribution (Single Variate Output)


    properties
        w;                % weight vector
        w0; % offset term
        df; % degrees of freedom
        sigma2;           % noise variance                          
        transformer;      % A data transformer object, e.g. KernelTransformer
        addOffset; 
    end

    %% Main methods
    methods
        function obj = Linreg(varargin)
          % Linreg(transformer, addOffset, w, w0, sigma2)
            [obj.transformer, obj.addOffset, obj.w,  obj.w0, obj.sigma2] = processArgs(varargin,...
                        '-transformer', [], ...
                        '-addOffset', true, ...
                        '-w'          , [], ...
                         '-w0'          , [], ...
                        '-sigma2'     , []);
        end
       
        function model = fit(model,D)
          % m = fit(m, D)
          % D is DataTable containing:
          % X(i,:) is i'th input; do *not* include a column of 1s
          % y(i) is i'th response
          %[D] = processArgs(varargin, '-D', []);
          X = D.X; y = D.Y; 
          if ~isempty(model.transformer)
            [X, model.transformer] = train(model.transformer, X);
          end
          n = size(X,1);
          if model.addOffset
            X = [X ones(n,1)];
          end
          w = X \ y; % least squares
          if model.addOffset
            model.w0 = w(end);
            model.w = w(1:end-1);
          else
            model.w0 = 0;
            model.w = w;
          end
          model.df = length(w);
          yhat = X*w;
          model.sigma2 = mean((yhat-y).^2);
        end

        function [yhat, py] = predict(model,X)
          %  X(i,:) is i'th input
          % yhat(i) = E[y | X(i,:)]
          % py(i) = p(y|X(i,:)), a GaussDist
          if ~isempty(model.transformer)
            X = test(model.transformer, X);
          end
          n = size(X,1);
          w0 = model.w0;
          ww = [model.w; w0];
          if isempty(model.w)
            X1 = ones(n,1);
          else
            X1 = [X ones(n,1)];
          end
          yhat = X1*ww;
          if nargout >= 2
            sigma2Hat = model.sigma2*ones(n,1); % constant variance!
            py = GaussDist(yhat, sigma2Hat);
          end
        end
  
        function model = mkRndParams(model, d)
         % Generate and set random d-dimensional parameters    
            model.w = randn(d,1);
            model.w0 = randn(1,1);
            model.sigma2 = rand(1,1);
        end

        function np = dof(model)
          np = model.df; % length(model.w);
        end
          
        function d = ndimensions(model)
          d = length(model.w);
        end
        
        function p = logprob(model, D)
          % D is DataTable containing X(i,:) and y(i)
          % p(i) = log p(y(i) | X(i,:), model params)
          X = D.X; y = D.Y; 
          yhat = predict(model, X);
          s2 = model.sigma2;
          p = -1/(2*s2)*(y(:)-yhat(:)).^2 - 0.5*log(2*pi*s2);
          if 0 % debug
            [yhat, py] = predict(model, X);
            pp = logprob(py, y);
            assert(approxeq(p, pp))
          end
        end
               
        function p = squaredErr(model, D)
          % p(i) = (y(i) - yhat(i))^2
          X = D.X; y = D.Y; 
          yhat = predict(model, X);
          p  = (y(:)-yhat(:)).^2;
        end

    end


   

end