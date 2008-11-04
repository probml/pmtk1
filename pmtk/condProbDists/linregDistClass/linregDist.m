classdef linregDist < condProbDist
  
  properties
    w; 
    sigma2;
    transformer;
  end
  
  %% Main methods
  methods 
    function m = linregDist(varargin)
      [transformer, w, sigma2] = process_options(...
        varargin, 'transformer', [],  'w', [], 'sigma2', []);
      m.transformer = transformer;
      m.w = w;
      m.sigma2 = sigma2;
    end
     
     function obj = mkRndParams(obj, d)
       obj.w = randn(d,1);
       obj.sigma2 = rand(1,1);
     end
    
     function p = logprob(obj, X, y)
       % p(i) = log p(Y(i,:) | X(i,:), params)
       [yhat] = mean(predict(obj, X));
       s2 = obj.sigma2;
       p = -1/(2*s2)*(y(:)-yhat(:)).^2 - 0.5*log(2*pi*s2);
       %[yhat, py] = predict(obj, X);
       %PP = logprob(py, y); % PP(i,j) = p(Y(i)| yhat(j))
       %p1 = diag(PP);
       %yhat = predict(obj, X);
       %assert(approxeq(p,p1))
     end

      function p = squaredErr(obj, X, y)
        yhat = mean(predict(obj, X));
        p  = (y(:)-yhat(:)).^2;
      end
     
      function py = predict(obj, X)
       if ~isempty(obj.transformer)
         X = test(obj.transformer, X);
       end
       n = size(X,1);
       muHat = X*obj.w;
       sigma2Hat = obj.sigma2*ones(n,1); % constant variance!
       py = gaussDist(muHat, sigma2Hat);
      end
      
      function [py] = postPredict(obj, X)
        if ~isempty(obj.transformer)
          X = test(obj.transformer, X);
        end
        n = size(X,1);
        done = false;
         switch class(obj.w)
          case 'mvnDist'
            if isa(obj.sigma2, 'double')
              muHat = X*obj.w.mu;
              Sn = obj.w.Sigma;
              sigma2Hat = obj.sigma2*ones(n,1) + diag(X*Sn*X');
              %{
              for i=1:n
                xi = X(i,:)';
                s2(i) = obj.sigma2 + xi'*Sn*xi;
              end
              assert(approxeq(sigma2Hat, s2))
              %}
              py = gaussDist(muHat, sigma2Hat);
              done = true;
            end
           case 'mvnInvGammaDist'
             wn = obj.w.mu; 
             Sn = obj.w.Sigma;
             vn = obj.w.a*2;
             sn2 = 2*obj.w.b/vn;
             m = size(X,n);
             SS = sn2*(eye(m) + X*Sn*X');
             py = studentDist(vn, X*wn, diag(SS));
             done = true;
         end
         assert(done)
      end
     
      function obj = inferParams(obj, varargin)
        % m = inferParams(model, 'name1', val1, 'name2', val2, ...)
        % Arguments are
        % 'X' - X(i,:) Do NOT include a column of 1's
        % 'y'- y(i)
        % lambda >= 0
        % 'prior' - one of {mvnDist object, mvnInvGammaDist object, ...
        %                   'mvn', 'mvnIG'}
        % In the latter 2 cases, we create a diagonal Gaussian prior
        % with precision lambda (except for the offset term)
        [X, y, lambda, prior, sigma2] = process_options(...
          varargin, 'X', [], 'y', [], 'lambda', 1e-3, 'prior', 'mvn',...
          'sigma2', []);
        if ~isempty(obj.transformer)
          [X, obj.transformer] = train(obj.transformer, X);
        end
        if isa(prior, 'char')
          obj.w = makeSphericalPrior(obj, X, lambda, prior);
        end
        if ~isempty(sigma2)
          % this is ignored if the prior is mvnIG
          obj.sigma2 = sigma2; 
        end
        done = false;
        switch class(obj.w)
            case 'mvnDist'
            if isa(obj.sigma2, 'double') && obj.sigma2 > 0
              % conjugate updating of w with fixed sigma2
              S0 = obj.w.Sigma; w0 = obj.w.mu;
              s2 = obj.sigma2; sigma = sqrt(s2);
              Lam0 = inv(S0); % yuck!
              [wn, Sn] = normalEqnsBayes(X, y, Lam0, w0, sigma);
              obj.w = mvnDist(wn, Sn);
              done = true;
            end
          case 'mvnInvGammaDist'
            % conjugate updating with unknown w and sigma2
            obj.w = updateMVNIG(obj, X, y);
           done = true;
        end
        assert(done)
      end
      
     function obj = fit(obj, varargin)
      % m = fit(model, 'name1', val1, 'name2', val2, ...)
      % Arguments are
      % 'X' - X(i,:) Do NOT include a column of 1's
      % 'y'- y(i)
      % 'prior' - one of {'none', 'L2', 'L1'}
      % 'lambda' >= 0
      % method - must be one of { ridgeQR, ridgeSVD }.
      [X, y, method, lambda, prior] = process_options(...
        varargin, 'X', [], 'y', [], 'method', 'ridgeQR', ...
        'lambda', 0, 'prior', 'none');
      if lambda>0 && strcmpi(prior, 'none'), prior = 'L2'; end
      if ~isempty(obj.transformer)
         [X, obj.transformer] = train(obj.transformer, X);
      end
      switch lower(prior)
        case 'none'
          n = size(X,1);
          obj.w = X \ y;
          yhat = X*obj.w;
          obj.sigma2 = mean((yhat-y).^2); % 1/n, not unbiased
          
        case 'l2'
          if ~isempty(obj.transformer) && addOffset(obj.transformer)
            X = X(:,2:end); % remove leading column of 1s
            addOnes = true;
          else
            addOnes = false;
          end
          obj.w = ridgereg(X, y, lambda, method, addOnes);
          n = size(X,1);
          if addOnes
            X = [ones(n,1) X]; % column of 1s for w0 term
          end
          yhat = X*obj.w;
          obj.sigma2 = mean((yhat-y).^2); % 1/n, not unbiased
          
        case 'l1'
          % lasso
          error('not yet implemented')
        otherwise
          error(['unrecognized method ' method])
      end
     end
    
     function s = bicScore(obj, X, y, lambda)
       L = sum(logprob(obj, X, y));
       n = size(X,1);
       %d = length(obj.w);
       d = dofRidge(obj, X, lambda);
       s = L-0.5*d*log(n);
     end
     
     function s = aicScore(obj, X, y, lambda)
       L = sum(logprob(obj, X, y));
       n = size(X,1);
       %d = length(obj.w);
       d = dofRidge(obj, X, lambda);
       s = L-d;
     end
     
     function df = dofRidge(obj, X, lambdas)
       % compute the degrees of freedom for a given lambda value
       % Elements of Statistical Learning p63
       if nargin < 3, lambdas = obj.lambda; end
       if ~isempty(obj.transformer)
         X = train(obj.transformer, X);
         if addOffset(obj.transformer)
           X = X(:,2:end);
         end
       end
       xbar = mean(X);
       XC = X - repmat(xbar,size(X,1),1);
       [U,D,V] = svd(XC,'econ');
       D2 = diag(D.^2);
       for i=1:length(lambdas)
         df(i) = sum(D2./(D2+lambdas(i)));
       end
     end

  end
  
end