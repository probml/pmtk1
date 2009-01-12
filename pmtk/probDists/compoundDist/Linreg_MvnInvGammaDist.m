classdef Linreg_MvnInvGammaDist < CondProbDist
%% Linear regression with MVNIG prior on w and sigma2


    properties
     wSigmaDist;
     transformer;   
     priorStrength;
    end

    %% Main methods
    methods
        function model = Linreg_MvnInvGammaDist(varargin)
            [model.transformer, model.wSigmaDist, model.priorStrength] = ...
              process_options(varargin,...
                        'transformer', []                      ,...      
                        'wSigmaDist'          , [], ....
                        'priorStrength', []);
        end

    
        function model = fit(model,varargin)
        % 'X'
        % 'y'
         % 'priorStrength' - magnitude of diagonals on precision matrix
        %    (defauly model.priorStrength); only used if model.wSigmaDist
        %    is []
         [X, y,  lambda] = process_options(varargin,...
          'X', [], 'y', [], 'priorStrength',model.priorStrength);
        if ~isempty(model.transformer)
          [X, model.transformer] = train(model.transformer, X);
        end
        if isempty(model.wSigmaDist) && ~isempty(lambda)
          d = size(X,2);
          model.wSigmaDist = makeSphericalPrior(d, lambda, addOffset(model.transformer), 'mvnig');
        end
        a0 = model.wSigmaDist.a;
        b0 = model.wSigmaDist.b;
        w0 = model.wSigmaDist.mu;
        S0 = model.wSigmaDist.Sigma;
        v0 = 2*a0; s02 = 2*b0/v0;
        d = length(w0);
        if det(S0)==0
          noninformative = true;
          Lam0 = zeros(d,d);
        else
          noninformative = false;
          Lam0 = inv(S0);
        end
        [wn, Sn] = normalEqnsBayes(X, y, Lam0, w0, 1);
        n = size(X,1);
        vn = v0 + n;
        an = vn/2;
        if noninformative
          sn2 = (1/vn)*(v0*s02 + (y-X*wn)'*(y-X*wn));
        else
          sn2 = (1/vn)*(v0*s02 + (y-X*wn)'*(y-X*wn) + (wn-w0)'*Sn*(wn-w0));
        end
        bn = vn*sn2/2;
        model.wSigmaDist = MvnInvGammaDist('mu', wn, 'Sigma', Sn, 'a', an, 'b', bn);
        end

         function py = predict(model,X)
          % py(i) = p(y|X(i,:), params), a GaussDist
          %X = process_options(varargin, 'X', []);
          if ~isempty(model.transformer)
            X = test(model.transformer, X);
          end
          wn = model.wSigmaDist.mu;
          Sn = model.wSigmaDist.Sigma;
          vn = model.wSigmaDist.a*2;
          sn2 = 2*model.wSigmaDist.b/vn;
          n = size(X,1); 
          SS = sn2*(eye(n) + X*Sn*X');
          py = StudentDist(vn, X*wn, diag(SS));
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
            model = Linreg_MvnInvGammaDist('transformer',T);
            model = fit(model,'X',Xtrain,'y',ytrain, 'prior', 'ridge', 'lambda', lambda);
            yp = predict(model,Xtest);
            err = mse(ytest, mode(yp))
        end 
    end

end