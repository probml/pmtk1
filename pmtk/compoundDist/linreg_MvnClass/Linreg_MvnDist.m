classdef Linreg_MvnDist < CondProbDist
%% Linear regression witn mutlivariate Gaussian prior on weights


    properties
     wDist;
     sigma2;
     transformer;      
     priorStrength;
    end

    %% Main methods
    methods
        function model = Linreg_MvnDist(varargin)
            [model.transformer, model.wDist, model.sigma2, model.priorStrength] = ...
              process_options(varargin,...
                        'transformer', []                      ,...      
                        'wDist'          , []                      ,... 
                        'sigma2', [], ....
                        'priorStrength', []);
        end

       
       
        function model = fit(model,varargin)
        % 'X'
        % 'y'
        % 'priorStrength' - magnitude of diagonals on precision matrix
        %    (defauly model.priorStrength); only used if model.wDist is []
         [X, y, lambda] = process_options(varargin,...
          'X', [], 'y', [], 'priorStrength', model.priorStrength);
        if ~isempty(model.transformer)
          [X, model.transformer] = train(model.transformer, X);
        end
        if isempty(model.wDist) && ~isempty(lambda)
          d = size(X,2);
          model.wDist = makeSphericalPrior(d, lambda, addOffset(model.transformer), 'mvn');
        end
        % conjugate updating of w with fixed sigma2
        S0 = model.wDist.Sigma; w0 = model.wDist.mu;
        s2 = model.sigma2; sigma = sqrt(s2);
        Lam0 = inv(S0); % yuck!
        [wn, Sn] = normalEqnsBayes(X, y, Lam0, w0, sigma);
        model.wDist = MvnDist(wn, Sn);
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
          muHat = X*model.wDist.mu;
          Sn = model.wDist.Sigma;
          sigma2Hat = model.sigma2*ones(n,1) + diag(X*Sn*X');
          py = GaussDist(muHat, sigma2Hat);
          %{
              for i=1:n
                xi = X(i,:)';
                s2(i) = model.sigma2 + xi'*Sn*xi;
              end
              assert(approxeq(sigma2Hat, s2))
          %}
         end
        
    end

   
    methods(Static = true)
        
         function testClass()
            load prostate;
            lambda = 0.05;
            T = ChainTransformer({StandardizeTransformer(false),AddOnesTransformer()});
            %XX = train(T, X);  d = size(XX,2);
            %prior = makeSphericalPrior(d, lambda, addOffset(T), 'mvnig');
            %model = Linreg_MvnInvGammaDist('wSigmaDist', prior, 'transformer',T);
            model = Linreg_MvnDist('transformer',T, 'priorStrength', lambda);
            model = fit(model,'X',Xtrain,'y',ytrain);
            yp = predict(model,'X',Xtest);
            err = mse(ytest, mode(yp))
         end 
        
       
    end

end