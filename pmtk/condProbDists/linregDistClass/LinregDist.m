classdef LinregDist < CondProbDist
%% Linear Regression Conditional Distribution (Single Variate Output)


    properties
        w;                   % w is the posterior distribution of the weights. 
                             % The form depends on how this object was fit. 
                             % If method = 'map', the default, then w 
                             % represents the MAP estimate and is stored as a 
                             % ConstDist object. If method = 'bayesian', then
                             % the form is conjugate to the prior, i.e. either
                             % an MvnDist object in the case that the variance,
                             % sigma2, is known, or an MvnInvGammaDist object in
                             % the case that both the mean and variance are
                             % unknown. 
                             
        sigma2;              % Fixed/Known value for the variance in 
                             % p(y | w,X) = N(y | Xw, sigma2*I). This value is
                             % ignored when an MvnInvGammaDist prior is
                             % specified. 
                             
        transformer;         % A data transformer object, e.g. KernelTransformer
    end

    %% Main methods
    methods
        function model = LinregDist(varargin)
        %Constructor
        %
        % FORMAT:
        %                     model = LinRegDist('name1',val1,'name2',val2,...);
        %       
        %
        % INPUT: 
        % 
        % 'transformer'     - An optional transformer object such as  KernelTransformer
        %
        % 'w'               - The posterior of the weights: either a ConstDist,
        %                     MvnDist or MvnInvGammaDist object. This will be
        %                     set automatically when fitting the object and can
        %                     be left unspecified. If w is a numeric matrix, it
        %                     is converted to a ConstDist object. 
        %
        % 'sigma2'          - Fixed/Known value for the variance in 
        %                     p(y | w,X) = N(y | Xw, sigma2*I). This is only used
        %                     when the fit method is 'bayesian' and the specified 
        %                     prior is an MvnDist object. When the object
        %                     is fit using method = MAP, sigma2 is estimated as
        %                     mean( (yhat-y).^2). It can also be specified
        %                     when fitting the model. 
        %
        % OUTPUT:           
        %
        % model             - The constructed LinregDist object
        %
        
            [transformer, w, sigma2] = process_options(varargin,...
                        'transformer', []                      ,...      
                        'w'          , []                      ,... 
                        'sigma2'     , []                      );
                    
            model.transformer = transformer;
            if(isnumeric(w) && ~isempty(w))
                model.w = ConstDist(w);
            else
                model.w = w;
            end
            model.sigma2 = sigma2;
        end

        function obj = set.w(obj,val)
            
            obj.w = val; 
            
        end
       
        function model = fit(model,varargin)
        %           model = fit(model,'name1',val1,'name2',val2,...)
        %
        % INPUT
        %
        % 'X'      - The training examples: X(i,:) is the ith case
        %
        % 'y'      - The output or target values s.t. y(i) corresponds to X(i,:)
        %
        % 'prior'  - In the case that method = 'map' estimation,(default), one
        %            of four strings can be specified, 'none' | 'L1' | 'L2' | 'L1L2' 
        %            'none' corresponds to the mle.
        %            'L1' corresponds to lasso,
        %            'L2' corresponds to ridge
        %            'L1L2' corresponds to elastic net. 
        %             This is the prior on weights w.
        %             Sigma is estimated by MLE (residual variance).
        %
        %            In the case that method = 'bayesian', you can specify a
        %            preconstructed MvnDist object or an MvnInvGammaDist
        %            object as the prior. If the former, sigma2 must be
        %            known. If the latter, the posterior p(w,sigma2|D) is
        %            estimated.
        %            Alternatively, you can specify the string 'mvn' or 'mvnIG'
        %            and an object will be constructed for you. In these latter 
        %            two cases, the prior is spherical with lambda used as 
        %            the precision, (except for the offset terms if any) and
        %            other values set to reasonable defaults.
        %
        % 'lambda'   In the case of map estimation, this is the regularization
        %            value. In the case of 'bayesian' estimation where the
        %            'prior' was specified as a string, this acts as the
        %            precision.
        %
        %            In the case of elastic net, lambda should be a vector of
        %            two elements: the L1 penalty followed by the L2 penalty. 
        % 
        % 'method'   One of ['map' | 'bayesian'] - if not specified, it is
        %            automatically infered from the type of the prior. 
        %
        % 'algorithm' Only used when method is 'map'
        %
        %            case: {'L1','L1L2'}  ['lars']    | 'shooting'  
        %            case: {'L2'}         ['ridgeQR'] | 'ridgeSVD'
        %               
        %                  
            [method,prior,remaining_args] = process_options(varargin,...
                                           'method' , 'default',...
                                           'prior'  , []);                      %#ok
            if(strcmp(method,'default'))
               if(isempty(prior))
                   method = 'mle';
               elseif(ischar(prior))
                   switch(lower(prior))
                       case {'l1','l2','l1l2'}
                           method = 'map';
                       case {'mvn','mvnig'}
                           method = 'bayesian';
                       otherwise
                           error('%s is an unsupported prior',prior);
                   end
               else
                   method = 'bayesian';
               end
            end
                                    
            switch lower(method)
                
                case {'map','mle'}
                   model = fitMAP(model,varargin{:});
                case 'bayesian'
                   model = fitBayesian(model,varargin{:});
                otherwise
                    error('%s is not supported - choose one of ''map'' or ''bayesian''',method);
            end
        end

        function py = predict(model, varargin)
        % Return a predictive distribution given the fitted model and the
        % test set examples, X. 
        %
        % FORMAT:
        %
        % py = predict(model,X)
        %
        % INPUT:
        % 
        % 'X'        - The test set examples, X(i,:) is example i
        % 
        % OUTPUT
        % py is a distribution such that py(i) = p(y|X(i,:),model) 
        % If the model parameters are a point estimate,
        % then py(i) = GaussDist(w' X(i,:), sigma2)
        % If the model params are a MVNIG distributionm
        % then py(i) = StudentDist(...) whose parameters depend on X(i,:)
        % If the model params are a MVN distribution with fixed sigma2.
        % then py(i) = GaussDist(E[w]' X(i,:), sigma2)
        
      

            if(nargin == 2 && isnumeric(varargin{1}))
                X = varargin{1};
            else
                X = process_options(varargin,'X',[]);
            end
        
        
            switch class(model.w)
                case {'MvnDist','MvnInvGammaDist'}
                    method = 'full';
                otherwise
                    method = 'plugin';
            end
            

            switch method

                case 'plugin'
                    if ~isempty(model.transformer)
                        X = test(model.transformer, X);
                    end
                    n = size(X,1);
                    m = mode(model.w);
                    muHat = X*m(:);
                    sigma2Hat = model.sigma2*ones(n,1); % constant variance!
                    py = GaussDist(muHat, sigma2Hat);

                case 'full'
                    py = predictBayesian(model,X);
                    
            end



        end
  
        function model = mkRndParams(model, d)
         % Generate and set random d-dimensional parameters    
            model.w = ConstDist(randn(d,1));
            model.sigma2 = rand(1,1);
        end

        function p = logprob(model, X, y)
        % p(i) = log p(y(i) | X(i,:), model params)
            [yhat] = mean(predict(model, X));
            s2 = model.sigma2;
            p = -1/(2*s2)*(y(:)-yhat(:)).^2 - 0.5*log(2*pi*s2);
            %[yhat, py] = predict(model, X);
            %PP = logprob(py, y); % PP(i,j) = p(Y(i)| yhat(j))
            %p1 = diag(PP);
            %yhat = predict(model, X);
            %assert(approxeq(p,p1))
        end

        function p = squaredErr(model, X, y)
        % Predict on X and compute the squared error between the specified y 
        % values and the predicted values yhat. 
            yhat = mean(predict(model, X));
            p  = (y(:)-yhat(:)).^2;
        end

        function s = bicScore(model, X, y, lambda)
        % Bayesian Information Criterion    
            L = sum(logprob(model, X, y));
            n = size(X,1);
            %d = length(model.w);
            d = dofRidge(model, X, lambda);
            s = L-0.5*d*log(n);
        end

        function s = aicScore(model, X, y, lambda)
        % Akaike Information Criterion    
            L = sum(logprob(model, X, y));
            d = dofRidge(model, X, lambda);
            s = L-d;
        end

        function df = dofRidge(model, X, lambdas)
        % Compute the degrees of freedom for a given lambda value
        % Elements of Statistical Learning p63
            if nargin < 3, lambdas = model.lambda; end
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

    %% Protected Methods
    methods(Access = 'protected')

        function model = fitMAP(model, varargin)
        % Helper method to fit perform MAP estimation     
        % Used by fit() when method = 'map'    
        % m = fit(model, 'name1', val1, 'name2', val2, ...)
        % Arguments are
        % 'X' - X(i,:) Do NOT include a column of 1's
        % 'y'- y(i)
        % 'prior' - one of {'none', 'L2', 'L1'}
        % 'lambda' >= 0
        % algorithm - must be one of { ridgeQR, ridgeSVD }.
            [X, y, algorithm, lambda, prior,method] = process_options(...
                varargin, 'X', [], 'y', [], 'algorithm', 'default', ...
                'lambda', 0, 'prior', 'none','method',[]);
            if all(lambda>0) && strcmpi(prior, 'none'), prior = 'L2'; end
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
                    model.w = ConstDist(X \ y);
                case 'l2'
                    if onesAdded
                        X = X(:,2:end); % remove leading column of 1s
                    end
                    model.w = ConstDist(ridgereg(X, y, lambda, algorithm, onesAdded));
                    n = size(X,1);
                    if onesAdded
                        X = [ones(n,1) X]; % column of 1s for w0 term
                    end
                case 'l1'
                    switch lower(algorithm)
                        case 'shooting'
                            model.w = ConstDist(LassoShooting(X,y,lambda,'offsetAdded',onesAdded));
                        case 'lars'
                            if(onesAdded)
                                w = larsLambda(X(:,2:end),center(y),lambda)';
                                w0 = mean(y)-mean(X)*(X\center(y));
                                model.w = ConstDist([w0;w]);
                            else
                                model.w = ConstDist(larsLambda(X,y,lambda)');
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
                    if(onesAdded)
                          w0 = mean(y)-mean(X)*(X\center(y));
                          w = [w0;w(:)];                                                                            %#ok
                          X = [ones(size(X,1),1),X];
                    end
                    model.w = ConstDist(w);
                otherwise
                    error(['unrecognized method ' method])
                  
            end
             yhat = X*model.w.point(:);
             model.sigma2 = mean((yhat-y).^2);
        end


        function model = fitBayesian(model, varargin)
        % Helper method to perform Bayesian inference    
        % Used by fit() when method = 'bayesian'    
        % m = fitBayesian(model, 'name1', val1, 'name2', val2, ...)
        % Arguments are
        % 'X' - X(i,:) Do NOT include a column of 1's
        % 'y'- y(i)
        % lambda >= 0
        % 'prior' - one of {MvnDist object, MvnInvGammaDist object, ...
        %                   'mvn', 'mvnIG'}
        % In the latter 2 cases, we create a diagonal Gaussian prior
        % with precision lambda (except for the offset term)
            [X, y, lambda, prior, sigma2,method] = process_options(...
                varargin, 'X', [], 'y', [], 'lambda', 1e-3, 'prior', 'mvn',...
                'sigma2', [],'method',[]);
            if ~isempty(model.transformer)
                [X, model.transformer] = train(model.transformer, X);
            end
            if isa(prior, 'char')
                model.w = makeSphericalPrior(model, X, lambda, prior);
            end
            if ~isempty(sigma2)
                % this is ignored if the prior is mvnIG
                model.sigma2 = sigma2;
            end
            done = false;
            switch class(model.w)
                case 'MvnDist'
                    if(isempty(model.sigma2))
                        error('You need to specify sigma2 when fitting with an MvnDist prior');
                    end
                    % conjugate updating of w with fixed sigma2
                    S0 = model.w.Sigma; w0 = model.w.mu;
                    s2 = model.sigma2; sigma = sqrt(s2);
                    Lam0 = inv(S0); % yuck!
                    [wn, Sn] = normalEqnsBayes(X, y, Lam0, w0, sigma);
                    model.w = MvnDist(wn, Sn);
                    done = true;
                    
                case 'MvnInvGammaDist'
                    % conjugate updating with unknown mu and sigma2
                    model.w = updateMVNIG(model, X, y);
                    done = true;
            end
            assert(done)
        end

        function [py] = predictBayesian(model, X)
          % Used by predict, when method = 'bayesian'
          if ~isempty(model.transformer)
            X = test(model.transformer, X);
          end
          n = size(X,1);
          done = false;
          switch class(model.w)
            case 'MvnDist'
              if isa(model.sigma2, 'double')
                muHat = X*model.w.mu;
                Sn = model.w.Sigma;
                sigma2Hat = model.sigma2*ones(n,1) + diag(X*Sn*X');
                %{
              for i=1:n
                xi = X(i,:)';
                s2(i) = model.sigma2 + xi'*Sn*xi;
              end
              assert(approxeq(sigma2Hat, s2))
                %}
                py = GaussDist(muHat, sigma2Hat);
                done = true;
              end
            case 'MvnInvGammaDist'
              wn = model.w.mu;
              Sn = model.w.Sigma;
              vn = model.w.a*2;
              sn2 = 2*model.w.b/vn;
              m = n; % size(X,n);
              SS = sn2*(eye(m) + X*Sn*X');
              py = StudentDist(vn, X*wn, diag(SS));
              done = true;
          end
          assert(done)
        end

    end
    
    methods(Static = true)
        
        function testClass()
            load prostate;
            lambda = 0.05;
            sigma2 = 0.5;
            T = ChainTransformer({StandardizeTransformer(false),AddOnesTransformer()});
            model = LinregDist('transformer',T);
            %% L1 shooting
            modelL1shooting = fit(model,'X',Xtrain,'y',ytrain,'method','map','prior','l1','lambda',lambda,'algorithm','shooting');
            yp = predict(modelL1shooting,'X',Xtest);
            L1shootingErr = mse(ytest,mode(yp))
            %% L1 lars
            modelL1lars = fit(model,'X',Xtrain,'y',ytrain,'method','map','prior','l1','lambda',lambda,'algorithm','lars');
            yp = predict(modelL1lars,'X',Xtest);
            L1larsErr = mse(ytest,mode(yp))
            %% L2 QR
            modelL2qr = fit(model,'X',Xtrain,'y',ytrain,'method','map','prior','l2','lambda',lambda,'algorithm','ridgeQR');
            yp = predict(modelL2qr,'X',Xtest);
            L2qrErr = mse(ytest,mode(yp))
            %% L2 SVD
            modelL2svd = fit(model,'X',Xtrain,'y',ytrain,'method','map','prior','l2','lambda',lambda,'algorithm','ridgeSVD');
            yp = predict(modelL2svd,'X',Xtest);
            L2svdErr = mse(ytest,mode(yp))
            %% Elastic Net, lars
            modelElasticLars = fit(model,'X',Xtrain,'y',ytrain,'method','map','prior','l1l2','lambda',[lambda,lambda],'algorithm','lars');
            yp = predict(modelElasticLars,'X',Xtest);
            elasticLarsErr = mse(ytest,mode(yp))
            %% Elastic Net shooting
            modelElasticShooting = fit(model,'X',Xtrain,'y',ytrain,'method','map','prior','l1l2','lambda',[lambda,lambda],'algorithm','shooting');
            yp = predict(modelElasticShooting,'X',Xtest);
            elasticShootingErr = mse(ytest,mode(yp))
            %% MLE
            modelMLE = fit(model,'X',Xtrain,'y',ytrain);
            yp = predict(modelMLE,Xtest);
            mleErr = mse(ytest,mode(yp))
            %% Bayesian MVN prior
            modelMVN = fit(model,'X',Xtrain,'y',ytrain,'method','bayesian','prior','mvn','lambda',lambda,'sigma2',sigma2);
            yp = predict(modelMVN,'X',Xtest);
            bayesMVNErr = mse(ytest, mode(yp))
            %% Bayesian MVNIG prior
            modelMVNIG = fit(model,'X',Xtrain,'y',ytrain,'method','bayesian','prior','mvnIG','lambda',lambda,'sigma2',sigma2);
            yp = predict(modelMVNIG,'X',Xtest);
            bayesMVNIGErr = mse(ytest,mode(yp))
        end 
    end

end