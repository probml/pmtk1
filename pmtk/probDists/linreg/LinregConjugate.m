classdef LinregConjugate < ProbDist
  %% Linear regression with conjugate prior
  % p(w,sigma2) = Mvn(w | 0, lambda*sigma2*I)  IG(sigma2 | a,b)
  % or, if sigma2 is fixed, just Mvn(w | 0, lambda*I)
 % The precision on the w0 (offset) term is 0
 
  properties
    wSigmaDist;
    wDist; sigma2;
    transformer;
    lambda;
  end

  %% Main methods
  methods
    function model = LinregConjugate(varargin)
      % LinregConjugate(transformer, lambda, sigma2, wDist, wSigmaDist)
      [model.transformer, model.lambda, model.sigma2, model.wDist, ...
        model.wSigmaDist] = ...
        processArgs(varargin,...
        '-transformer', [], ...
        '-lambda', 0.1, ...
        '-sigma2', [], ...
        '-wDist', [], ...
        '-wSigmaDist', []);
    end

    function model = fit(model,D)
      % m = fit(m, D), D is DataTable
      % D.X(i,:) is i'th input, do *not* include column of 1s
      % D.y(i)
      X = D.X; y = D.Y;
      [n,d] = size(X);
      if ~isempty(model.transformer)
        if addOffset(model.transformer)
        error('don''t include column of 1s')
        end
        [X, model.transformer] = train(model.transformer, X);
      end
      X = [ones(n,1) X]; 
      if ~isempty(model.sigma2)
        model = fitKnownVariance(model, X, y);
      else
        model = fitUnknownVariance(model, X, y);
      end
    end


    function [yhat, py] = predict(model,X)
      % py(i) = p(y|X(i,:)) is a Gaussian or StudentDist
      n = size(X,1);
      if ~isempty(model.transformer)
        X = test(model.transformer, X);
      end
      X = [ones(n,1) X];
      computeProbY = (nargout >= 2);
      if ~isempty(model.sigma2)
        [yhat,py] = predictKnownVariance(model, X, computeProbY);
      else
        [yhat,py] = predictUnknownVariance(model, X, computeProbY);
      end
    end
    
     function p = logprob(model,D)
      % p(i) = log p(D.y(i)|D.X(i,:)) log *marginal* likelihood
      X = D.X; y = D.Y;
      py = predict(model, X);
      p = logprob(py, y);
    end
   

  end % methods

  methods(Access = 'protected')
    
    function [yhat,py] = predictUnknownVariance(model, X, computeProbY)
      [n,d] = size(X);
      [wn,Sn,an,bn] = getHyperparamsUnknownVariance(model, d);
      yhat = X*wn;
      if ~computeProbY
        py = [];
      else
        vn = an*2; sn2 = 2*bn/vn;
        n = size(X,1);
        SS = sn2*(eye(n) + X*Sn*X');
        py = StudentDist(vn*ones(n,1), yhat, diag(SS));
      end
    end
    
    function [yhat, py] = predictKnownVariance(model, X, computeProbY)
      [n,d] = size(X);
      [wn,Sn] = getHyperparamsKnownVariance(model, d);
      yhat = X*wn;
      if ~computeProbY
        py = [];
      else
        sigma2Hat = model.sigma2*ones(n,1) + diag(X*Sn*X');
        py = GaussDist(yhat, sigma2Hat);
      end
    end

    function [w0,S0,a0,b0] = getHyperparamsUnknownVariance(model, d)
      if isempty(model.wSigmaDist) % make a prior of specified strength
        w0 = zeros(d,1);
        prior_precision = model.lambda*eye(d);
        prior_precision(1,1) = 1e-10; % for offset
        S0 = diag(1./diag(prior_precision));  % prior cov mat
        a0 = 0.01; b0 = 0.01;  % vague
      else
        a0 = model.wSigmaDist.a;
        b0 = model.wSigmaDist.b;
        w0 = model.wSigmaDist.mu(:);
        S0 = model.wSigmaDist.Sigma;
      end
    end
    
    function [w0,S0] = getHyperparamsKnownVariance(model, d)
      if isempty(model.wDist)
        w0 = zeros(d,1);
        prior_precision = model.lambda*eye(d);
        prior_precision(1,1) = 1e-10; % for offset
        S0 = diag(1./diag(prior_precision));  % prior cov mat
      else
        w0 = model.wDist.mu(:);
        S0 = model.wDist.Sigma;
      end
    end
    
    function model = fitUnknownVariance(model,X,y)
      d = size(X,2);
      [w0,S0,a0,b0] = getHyperparamsUnknownVariance(model, d);
      v0 = 2*a0; s02 = 2*b0/v0;
      d = length(w0);
      if det(S0)==0
        noninformative = true;
        Lam0 = zeros(d,d);
      else
        noninformative = false;
        Lam0 = inv(S0);
      end
      [wn, Sn] = LinregConjugate.normalEqnsBayes(X, y, Lam0, w0, 1);
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

   

    function model = fitKnownVariance(model,X, y)
      d = size(X,2);
      [w0,S0] = getHyperparamsKnownVariance(model, d);
      s2 = model.sigma2; sigma = sqrt(s2);
      Lam0 = inv(S0); % yuck!
      [wn, Sn] = LinregConjugate.normalEqnsBayes(X, y, Lam0, w0, sigma);
      model.wDist = MvnDist(wn, Sn);
    end

  end % protected


  methods(Static = true)

    function [wn, Sn] = normalEqnsBayes(X, y, Lam0, w0, sigma)
      % numerically stable solution to posterior mean and covariance
      [Lam0root, p] = chol(Lam0);
      if p>0
        d=length(w0);
        Lam0root = zeros(d,d);
      end
      Xtilde = [X/sigma; Lam0root];
      ytilde = [y/sigma; Lam0root*w0];
      [Q,R] = qr(Xtilde, 0);
      wn = R\(Q'*ytilde);
      if nargout >= 2
        Rinv = inv(R);
        Sn = Rinv*Rinv';
      end
      if false % naive way, for debugging
        s2 = sigma^2;
        Sninv = Lam0 + (1/s2)*(X'*X);
        Sn2 = inv(Sninv);
        wn2 = Sn2*(Lam0*w0 + (1/s2)*X'*y);
        assert(approxeq(Sn,Sn2))
        assert(approxeq(wn,wn2))
      end
    end % normalEqns

  end % Static

end