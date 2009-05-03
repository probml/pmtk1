classdef LinregL2 < Linreg
%% Ridge Regression  (Single Variate Output) usign QR decomposition


    properties
        lambda;
        Xtrain; % store data to compute dof later
    end
 
  
    %% Main methods
    methods
      function obj = LinregL2(varargin)
        % m = LinregL2(lambda, transfomer, w, w0, sigma2,  df, addOffset)
        [obj.lambda, obj.transformer, obj.w, obj.w0, obj.sigma2, obj.df, ...
          obj.addOffset] = ...
          processArgs(varargin,...
          '-lambda'      , 0, ...
          '-transformer', [], ...                 
          '-w'          , [], ...   
          '-w0'          , [], ...   
          '-sigma2'     , [], ...                     
          '-df', [], ...
          '-addOffset', true);
      end
       
        
        function df = dof(model)
          % slow since need evals of X
         df = LinregL2.dofRidge(model.Xtrain, model.lambda);
        end
        
    end % methods

    methods(Access = 'protected')
      
      function [w, out, model] = fitCore(model, XC, yC)
        d = size(XC,2);
        XX  = [XC; sqrt(model.lambda)*eye(d)];
        yy = [yC; zeros(d,1)];
        w  = XX \ yy; % QR
        out = [];
        model.Xtrain = XC; % for dof computation
      end
      
    end
        
    methods(Static = true)
      
      function df = dofRidge(X, lambdas)
        % Compute the degrees of freedom for a given lambda value
        % Elements1e p63
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

