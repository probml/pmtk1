classdef Linreg < ProbDist
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
            [obj.transformer, obj.addOffset, obj.w,  obj.w0, ...
              obj.sigma2] = processArgs(varargin,...
                        '-transformer', [], ...
                        '-addOffset', true, ...
                        '-w'          , [], ...
                         '-w0'          , [], ...
                        '-sigma2'     , []);
        end
       
        function [model, output] = fit(model,D)
          % m = fit(m, D)
          % D is DataTable containing:
          % X(i,:) is i'th input; do *not* include a column of 1s
          % y(i) is i'th response
          %[D] = processArgs(varargin, '-D', []);
          X = D.X; y = D.Y; 
          if ~isempty(model.transformer)
            [X, model.transformer] = train(model.transformer, X);
          end
           [d] = size(X,2);
          n = length(y);
          if d==0 % no inputs
            w0 = mean(y);
            w = [];
            output = [];
          else
            [XC, xbar] = center(X);
            [yC, ybar] = center(y);
            [w, output, model] = fitCore(model, XC, yC);
            w0 = ybar - xbar*w;
          end
          model.w = w; model.w0 = w0;
          if model.addOffset
            ww = [w0; w(:)];
            X = [ones(n,1) X]; 
          else
            ww = w(:);
          end
          yhat = X*ww;
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
          if model.addOffset
            X = [ones(n,1) X];
            ww = [model.w0; model.w];
          else
            ww = model.w;
          end
          yhat = X*ww;
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
          np = length(model.w);
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
        
    end % methods

    methods(Access = 'protected')
      
      function [w, out, model] = fitCore(model, XC, yC) %# ok
        w = XC \ yC; % least squares
        out = [];
      end
      
    end
    

end