classdef LinregL2ModelList < ModelList
%% Ridge Regression  for multiple lambda values using SVD for speed

    properties
        lambdas;
        nlambdas;
        transformer;
        addOffset = true;
    end
 
    %% Main methods
    methods
      function ML = LinregL2ModelList(varargin)
        % m = LinregL2ModelList(lambdas, nlambdas, transformer, selMethod, predMethod, nfolds,
        % verbose) 
        % eg m = LinregL2ModelList( '-nlambdas', 10)
        % See ModelList for explanation of arguments
        [ML.lambdas,  ML.nlambdas, ML.transformer, ML.selMethod, ML.predMethod, ...
          ML.nfolds, ML.verbose, ML.costFnForCV] = ...
          processArgs(varargin,...
          '-lambdas', [], ...
          '-nlambdas', [], ...
          '-transformer', [], ...
          '-selMethod', 'bic', ...
          '-predMethod', 'plugin', ...
          '-nfolds', 5, ...
          '-verbose', false, ...
           '-costFnForCV', (@(M,D) -logprob(M,D)) ...
           );
         if  ~isempty(ML.nlambdas)
          ML.lambdas = []; % will auto-generate once we see X,y
        end
        
      end
       
       function [W, w0, sigma2, dof, lambda] = getParamsForAllModels(ML)
        % Extract parameters from models in list and convert to matrix
        % W(:,m) = weights for model m
        % w0(m)
        % sigma2(m)
        % dof(m) = degrees of freedom
        % lambda(m)
        Nm = length(ML.models);
        d = ndimensions(ML.models{1});
        W = zeros(d,Nm); w0 = zeros(1,Nm); sigma2 = zeros(1,Nm);
        dof = zeros(1,Nm);  lambda = zeros(1,Nm);
        for m=1:Nm
          W(:,m) = ML.models{m}.w(:);
          w0(m) = ML.models{m}.w0;
          sigma2(m) = ML.models{m}.sigma2;
          dof(m) = ML.models{m}.df;
          lambda(m) = ML.models{m}.lambda;
        end
       end
      
    
      
      function [models] = fitManyModels(ML, D)
        X = D.X; y = D.Y; clear D;
        if isempty(ML.lambdas)
          ML.lambdas = [0 logspace(0, log(LinregL2ModelList.maxLambdaLinregL2(X)), ML.nlambdas-1)];
        end
        lambdas = ML.lambdas;
          if ~isempty(ML.transformer)
            [X, model.transformer] = train(ML.transformer, X);
          end
          [XC, xbar] = center(X);
          [yC, ybar] = center(y);
          
          [U,D,V] = svd(XC,'econ');
          D2 = diag(D.^2);
          [n,d] = size(X);  
          Nm = length(lambdas);
          models = cell(1, Nm);
          for i=1:Nm
            lambda = lambdas(i); %#ok
            if lambda==0
              w = pinv(XC)*yC;
            else
              w  = V*diag(1./(D2 + lambda))*D*U'*yC;
            end
            df = sum(D2./(D2+lambda));
            if ML.addOffset
              w0 = ybar - xbar*w; 
              ww = [w0; w(:)];
              ypred = [ones(n,1) X]*ww(:);
            else
              w0 = 0;
              ypred = X*w(:);
            end
            sigma2  = mean((y-ypred).^2);
            models{i} = LinregL2('-transformer', ML.transformer, '-w', w, ...
              '-w0', w0, '-addOffset', ML.addOffset, ...
              '-sigma2', sigma2, '-df', df, '-lambda', lambda);
          end
      end % fitManyModels

    end % methods

    methods(Static = true)
      
      function lambda = maxLambdaLinregL2(X)
        % auto-generate a reasonable range of lambdas
        % Obviously 0 is the minimum
        % We set the max to be the largest squared singular value
        XC  = center(X);
        [n,d] = size(XC);
        D22 = eig(XC'*XC); % evals of X'X = svals^2 of X
        D22 = sort(D22, 'descend');
        D22 = D22(1:min(n,d));
        lambda= 1*max(D22);
        if 0 % debug - svd slower than computing evals
          %X = rand(10,20);
          [U,D,V] = svd(XC,'econ'); % D2(i) = singular value
          D2 = diag(D.^2);
          assert(approxeq(D2,D22))
        end
      end
        
    end % Static
end % class
