classdef LinregL1ModelList < ModelList
%% Lasso Regression  for multiple lambda values using LARS 

    properties
        lambdas;
        nlambdas;
        transformer;
        debias;
        addOffset = true;
    end
 
    %% Main methods
    methods
      function ML = LinregL1ModelList(varargin)
        % m = LinregL1ModelList(lambdas, nlambdas, transformer, selMethod, predMethod, nfolds,
        % verbose, debias, costFnForCV) 
        % eg m = LinregL1ModelList('-nlambdas', 5) 
        % eg m = LinregL1ModelList('-lambdas', 'all') computes full reg path 
        % De-bias means we fit the params by least squares after identifying
        % support sets.
        % See ModelList for explanation of arguments
        [ML.lambdas, ML.nlambdas, ML.transformer, ML.selMethod, ML.predMethod, ...
          ML.nfolds, ML.verbose, ML.debias, ML.costFnForCV] = ...
          processArgs(varargin,...
          '-lambdas', 'all', ...
          '-nlambdas', [], ...
          '-transformer', [], ...
          '-selMethod', 'bic', ...
          '-predMethod', 'plugin', ...
          '-nfolds', 5, ...
          '-verbose', false, ...
          '-debias', false, ...
           '-costFnForCV', (@(M,D) -logprob(M,D)) ...
           );
        if  ~isempty(ML.nlambdas)
          ML.lambdas = []; % will auto-generate once we see X,y
        end
      end
       
      function [W, w0, sigma2, dof, lambda, shrinkage] = getParamsForAllModels(ML)
        % Extract parameters from models in list and convert to matrix
        % W(:,m) = weights for model m
        % w0(m)
        % sigma2(m)
        % dof(m) = degrees of freedom
        % lambda(m) = lambda for model m
        % srhinkage(m) = norm(W(:,m))/maxnorm
        Nm = length(ML.models);
        %d = ndimensions(ML.models{1});
        d = length(ML.models{1}.w);
        W = zeros(d,Nm); w0 = zeros(1,Nm); sigma2 = zeros(1,Nm);
        dof = zeros(1,Nm);  lambda = zeros(1,Nm);
        for m=1:Nm
          W(:,m) = ML.models{m}.w(:);
          w0(m) = ML.models{m}.w0;
          sigma2(m) = ML.models{m}.sigma2;
          dof(m) = ML.models{m}.df;
          lambda(m) = ML.models{m}.lambda;
        end
        if nargout >= 6
          %shrinkage =zeros(1,Nm);
          %wls = X\y; nw = norm(wls);
          for m=1:Nm
            t(m) = norm(W(:,m));
            %shrinkage(m) = norm(W(:,m)) / nw;
          end
          shrinkage = t/max(t); % guaranteed that max has shrinkage of 1
        end
      end
      
    end
    
    methods(Access = 'protected')
      function [models] = fitManyModels(ML, D)
        % We use lars to find the supports
        % but then optionally fit an 'unbiased' estimate of the weights
        X = D.X; y = D.Y; 
        if ~isempty(ML.transformer)
          [X, model.transformer] = train(ML.transformer, X);
        end
        [XC, xbar] = center(X);
        %XC = mkUnitVariance(XC);
        [yC, ybar] = center(y);
        if isempty(ML.lambdas) 
          ML.lambdas = linspace(0, lambdaMaxLasso(XC,yC), ML.nlambdas);
          %ML.lambdas = [0 exp(logspace(0, log(lambdaMaxLasso(XC,yC)), ML.nlambdas-1))];
        end
        lambdas = ML.lambdas;
        if ischar(lambdas) && strcmp(lambdas, 'all') 
          W = lars(XC, yC, 'lasso'); 
          lambdas = recoverLambdaFromLarsWeights(X,y,W);
          %lambdas = recoverLambdaFromLarsWeights(XC,yC,W);
        else
          W = larsLambda(XC,yC,lambdas);
        end
        %W(i,:) corresponds to the solution given lambdas(i)
        [Nm,d] = size(W);
        models = cell(1,Nm);
        for m=1:Nm
          ndx = find(abs(W(m,:)) ~= 0);
          df = length(ndx);
          if ML.debias
            % refit using ridge (for stability) on chosen subset
            tmp = LinregL2('-transformer', ML.transformer, '-lambda', 0.001, '-addOffset', true);
            tmp = fit(tmp, X(:, ndx), y);
            w = zeros(d,1); w(ndx) = tmp.w; w0 = tmp.w0;
          else
            w = W(m,:)'; 
            w0 = ybar - xbar*w;
          end
          ww = [w0; w(:)];
          n = length(y);
          ypred = [ones(n,1) X]*ww(:);
          sigma2  = mean((y-ypred).^2);
          models{m} = LinregL1('-transformer', ML.transformer, '-w', w, ...
            '-w0', w0, '-addOffset', ML.addOffset, ...
            '-sigma2', sigma2, '-df', df, '-lambda', lambdas(m)); %#ok
        end % for m
    end % fitManyModels

    end % methods

    methods(Static = true)
     
    end % Static
end % class
