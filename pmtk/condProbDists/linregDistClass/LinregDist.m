classdef LinregDist < CondProbDist
%% Linear Regression Conditional Distribution (Single Variate Output)


    properties
        w;                % weight vector
        sigma2; % noise variance                          
        transformer;         % A data transformer object, e.g. KernelTransformer
        prior; % for MAP estimation
    end

    %% Main methods
    methods
        function obj = LinregDist(varargin)
          % Optional args:
          % 'w'
          % 'sigma2'
          % 'transformer'
          % 'prior'
            [obj.transformer, obj.w, obj.sigma2, obj.prior] = process_options(varargin,...
                        'transformer', []                      ,...      
                        'w'          , []                      ,... 
                        'sigma2'     , []                      , ....
                        'prior'      , 'none');
        end
       
        function model = fit(model,varargin)
        %           model = fit(model,'name1',val1,'name2',val2,...)
        %
        % INPUT
        %
        % 'X'      - The training examples: X(i,:) is the ith case.
        %            Do not include a column of 1s.
        %
        % 'y'      - The output or target values s.t. y(i) corresponds to X(i,:)
        %
        % 'prior'  -  a string from {'none' | 'L1' | 'L2' | 'L1L2'}
        %            'none' corresponds to the mle.
        %            'L1' corresponds to lasso,
        %            'L2' corresponds to ridge
        %            'L1L2' corresponds to elastic net. 
        %             This is the prior on weights w.
        %             Sigma is estimated by MLE (residual variance).
        %
        % 'lambda'   Strenght of regularization/ penalty.
        %            In the case of elastic net, lambda should be a vector of
        %            two elements: the L1 penalty followed by the L2
        %            penalty. 
        %
        % 'algorithm' Only used when prior is 'L1', 'L2', or 'L1L2'
        %
        %            case: {'L1','L1L2'}  ['lars']    | 'shooting'  
        %            case: {'L2'}         ['ridgeQR'] | 'ridgeSVD'
        %               
        %                  
        [X, y, prior, lambda, algorithm] = process_options(varargin,...
          'X', [], 'y', [], 'prior', model.prior, 'lambda', [], 'algorithm', 'default');
        if(strcmp(algorithm,'default'))
          switch lower(prior)
            case {'l1','l1l2'}
              algorithm = 'lars';
            case 'l2'
              algorithm = 'ridgeQR';
          end
        end
        if ~isempty(model.transformer)
          [X, model.transformer] = train(model.transformer, X);
        end
        onesAdded = ~isempty(model.transformer) && addOffset(model.transformer);
        switch lower(prior)
          case 'none'
            model.w = X \ y;
          case 'l2'
            if onesAdded
              X = X(:,2:end); % remove leading column of 1s
            end
            model.w = ridgereg(X, y, lambda, algorithm, onesAdded);
            n = size(X,1);
            if onesAdded
              X = [ones(n,1) X]; % column of 1s for w0 term
            end
          case 'l1'
            switch lower(algorithm)
              case 'shooting'
                model.w = LassoShooting(X,y,lambda,'offsetAdded',onesAdded);
              case 'lars'
                if(onesAdded)
                  w = larsLambda(X(:,2:end),center(y),lambda)';
                  w0 = mean(y)-mean(X)*(X\center(y));
                  model.w = [w0;w];
                else
                  model.w = larsLambda(X,y,lambda)';
                end
              otherwise
                error('%s is not a supported L1 algorithm',algorithm);
            end
          case 'l1l2'
            % elastic net
            if(numel(lambda) ~= 2)
              error('Please specify two values for lambda, [lambdaL1, lambdaL2]');
            end
            if(onesAdded)
              X = X(:,2:end);
            end
            switch lower(algorithm)
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
          otherwise
            error(['unrecognized method ' prior])
        end % switch(prior)
        yhat = X*model.w;
        model.sigma2 = mean((yhat-y).^2);
        model.ndimsX = size(X,2);
        model.ndimsY = size(y,2);
        end

        function py = predict(model,X)
          % Input args:
          % 'X' - X(i,:) is i'th example
          %
          % Output
          % py(i) = p(y|X(i,:), params), a GaussDist
          %X = process_options(varargin, 'X', []);
          if ~isempty(model.transformer)
            X = test(model.transformer, X);
          end
          n = size(X,1);
          muHat = (X*model.w);
          sigma2Hat = model.sigma2*ones(n,1); % constant variance!
          py = GaussDist(muHat, sigma2Hat);
        end
  
        function model = mkRndParams(model, d)
         % Generate and set random d-dimensional parameters    
            model.w = randn(d,1);
            model.sigma2 = rand(1,1);
        end

       
        function s = bicScore(model, X, y, lambda)
        % Bayesian Information Criterion, assuming L2 prior
            L = sum(logprob(model, X, y));
            n = size(X,1);
            %d = length(model.w);
            d = dofRidge(model, X, lambda);
            s = L-0.5*d*log(n);
        end

        function s = aicScore(model, X, y, lambda)
        % Akaike Information Criterion, assuming L1 prior  
            L = sum(logprob(model, X, y));
            d = dofRidge(model, X, lambda);
            s = L-d;
        end

        function df = dofRidge(model, X, lambdas)
        % Compute the degrees of freedom for a given lambda value
        % Elements of Statistical Learning p63
        if ~isempty(model.transformer)
          X = train(model.transformer, X);
          if addOffset(model.transformer)
            X = X(:,2:end);
          end
        end
        xbar = mean(X);
        XC = X - repmat(xbar,size(X,1),1);
        [U,D,V] = svd(XC,'econ');                                           %#ok
        D2 = diag(D.^2);
        nlambdas = length(lambdas);
        df = zeros(nlambdas,1);
        for i=1:nlambdas
          df(i) = sum(D2./(D2+lambdas(i)));
        end
        end

    end


    methods(Static = true)
        
        function testClass()
            load prostate;
            lambda = 0.05;
            sigma2 = 0.5;
            T = ChainTransformer({StandardizeTransformer(false),AddOnesTransformer()});
            model = LinregDist('transformer',T);
            % L1 shooting
            modelL1shooting = fit(model,'X',Xtrain,'y',ytrain,'prior','l1','lambda',lambda,'algorithm','shooting');
            yp = predict(modelL1shooting,Xtest);
            L1shootingErr = mse(ytest,mode(yp))
            % L1 lars
            modelL1lars = fit(model,'X',Xtrain,'y',ytrain,'prior','l1','lambda',lambda,'algorithm','lars');
            yp = predict(modelL1lars,Xtest);
            L1larsErr = mse(ytest,mode(yp))
            % L2 QR
            modelL2qr = fit(model,'X',Xtrain,'y',ytrain,'prior','l2','lambda',lambda,'algorithm','ridgeQR');
            yp = predict(modelL2qr,Xtest);
            L2qrErr = mse(ytest,mode(yp))
            % L2 SVD
            modelL2svd = fit(model,'X',Xtrain,'y',ytrain,'prior','l2','lambda',lambda,'algorithm','ridgeSVD');
            yp = predict(modelL2svd,Xtest);
            L2svdErr = mse(ytest,mode(yp))
            % Elastic Net, lars
            modelElasticLars = fit(model,'X',Xtrain,'y',ytrain,'prior','l1l2','lambda',[lambda,lambda],'algorithm','lars');
            yp = predict(modelElasticLars,Xtest);
            elasticLarsErr = mse(ytest,mode(yp))
            % Elastic Net shooting
            modelElasticShooting = fit(model,'X',Xtrain,'y',ytrain,'prior','l1l2','lambda',[lambda,lambda],'algorithm','shooting');
            yp = predict(modelElasticShooting,Xtest);
            elasticShootingErr = mse(ytest,mode(yp))
            % MLE
            modelMLE = fit(model,'X',Xtrain,'y',ytrain);
            yp = predict(modelMLE,Xtest);
            mleErr = mse(ytest,mode(yp))
        end 
    end

end