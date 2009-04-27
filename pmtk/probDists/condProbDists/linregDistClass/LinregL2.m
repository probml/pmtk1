classdef LinregL2 < Linreg
%% Ridge Regression  (Single Variate Output) usign QR decomposition


    properties
        lambda;
        method;
    end
 
  
    %% Main methods
    methods
      function obj = LinregL2(varargin)
        % m = LinregL2(lambda, transfomer, w, w0, sigma2, method, dof)
        % method is one of {'ridgeQR', 'ridgeSVD'}
        [obj.lambda, obj.transformer, obj.w, obj.w0, obj.sigma2, obj.method, obj.df, obj.addOffset] = ...
          processArgs(varargin,...
          '-lambda'      , 0, ...
          '-transformer', [], ...                 
          '-w'          , [], ...   
          '-w0'          , [], ...   
          '-sigma2'     , [], ...                     
          '-method', 'ridgeQR', ...
          '-df', 0, ...
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
          %[model.w, model.w0] = ridgereg(X, y, model.lambda, model.method, model.addOffset);
        
          [n,d] = size(X);
          n = length(y);
          if d==0 % no inputs
            w0 = mean(y);
            w = [];
          else
            [XC, xbar] = center(X);
            [yC, ybar] = center(y);
            if model.lambda==0
              w = XC \ yC; % least squares
            else
              d = size(XC,2);
              XX  = [XC; sqrt(model.lambda)*eye(d)];
              yy = [yC; zeros(d,1)];
              w  = XX \ yy; % QR
            end
            w0 = ybar - xbar*w;
            if ~model.addOffset, w0 = 0; end
          end
          model.w = w; model.w0 = w0;
          model.df = LinregL2.dofRidge(X, model.lambda);
          ww = [w(:); w0];
          X1 = [X ones(n,1)]; % column of 1s for w0 term
          yhat = X1*ww;
          model.sigma2 = mean((yhat-y).^2);
          model.ndimsX = size(X,2);
          model.ndimsY = size(y,2);
        end

    end % methods

    methods(Static = true)
      function df = dofRidge(X, lambdas)
        % Compute the degrees of freedom for a given lambda value
        % Elements of Statistical Learning p63
        [n,d] = size(X);
        if d==0, df = 0; return; end
        XC  = center(X);
        D22 = eig(XC'*XC); % evals of X'X = svals^2 of X
        D22 = sort(D22, 'descend');
        D22 = D22(1:min(n,d));
        %[U,D,V] = svd(XC,'econ');                                           %#ok
        %D2 = diag(D.^2);
        %assert(approxeq(D2,D22))
        D2 = D22;
        nlambdas = length(lambdas);
        df = zeros(nlambdas,1);
        for i=1:nlambdas
          df(i) = sum(D2./(D2+lambdas(i)));
        end
      end
    end % methods

end % class

