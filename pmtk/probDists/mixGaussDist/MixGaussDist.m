classdef MixGaussDist < ParamDist 
  % mixture of Gaussians 
  
  properties
    mu;
    Sigma;
    K;
    mixweights;
    paramInfEng;
  end
  
  %% main methods
  methods
    function m = MixGaussDist(varargin)
      [K, mu, Sigma, mixweights] = process_options(varargin, ...
        'K', [], 'mu', [], 'Sigma', [], 'mixweights', []);
     m.K = K; m.mu = mu; m.Sigma = Sigma; m.mixweights = mixweights;
    end

    function objScalar = convertToScalarDist(obj);
      objScalar = obj;
    end
    
    function obj = mkRndParams(obj, d, K)
      obj.mu = randn(d,K);
      for k=1:K
        obj.Sigma(:,:,k) = randpd(d);
      end
      obj.mixweights = normalise(rand(1,K));
    end
    
    function d = ndims(m)
      d = size(m.mu,1);
    end
    
    function L = logprob(obj, X)
      % L(i) = log p(X(i,:) | params)
      n = size(X,1);
      LK = zeros(n, obj.K);
      for k=1:obj.K
        m = MvnDist(obj.mu(:,k), obj.Sigma(:,:,k));
        LK(:,k) = logprob(m, X) + repmat(log(obj.mixweights(k)), n, 1);
      end
      % L(i) = log sum_k exp[ log pi_k + log N(X(i,:) | mu(k), Sigma(k)) ]
      L = logsumexp(LK,2);
    end
       
  
    function mu = mean(m)
      checkParamsAreConst(m)
      d = ndims(m);
      %M = obj.mu .* repmat(obj.mixweights,d,1);
      M = bsxfun(@times,  m.mu, m.mixweights(:)');
      mu = sum(M, 2);
    end
    
    function C = cov(m)
      mu = mean(m);
      C = mu*mu';
      for k=1:m.K
        C = C + m.mixweights(k)*(m.Sigma(:,:,k) + m.mu(:,k)*m.mu(:,k)');
      end
    end
   
     function X = sample(obj, n)
      % X(i,:) = sample for i=1:n
      if nargin < 2, n = 1; end
      Z = sampleDiscrete(obj.mixweights, n, 1);
      d = ndims(obj);
      X = zeros(n, d);
      for k=1:obj.K
        ndx = find(Z==k);
        X(ndx,:) = mvnrnd(obj.mu(:,k)', obj.Sigma(:,:,k), length(ndx));
      end
     end
    
 
    
    function obj = fit(obj, varargin)
      % Point estimate of parameters
      % m = fit(model, 'name1', val1, 'name2', val2, ...)
      % Arguments are
      % data - data(i,:) = case i
      % suffStat -
      % method - one of {map, mle, covshrink}
      %
      % For covshrink: we use the Ledoit-Wolf formula to estimate srhinkage amount
      %  See  J. Schaefer and K. Strimmer.  2005.  A shrinkage approach to
      %   large-scale covariance matrix estimation and implications
      %   for functional genomics. Statist. Appl. Genet. Mol. Biol. 4:32.

      [X, SS, method] = process_options(...
        varargin, 'data', [], 'suffStat', [], 'method', 'mle');
      hasMissingData =  any(isnan(X(:)));
      assert(~hasMissingData)
      if isempty(SS), SS = MvnDist.mkSuffStat(X); end
      switch method
        case 'mle'
          obj.mu = SS.xbar;
          obj.Sigma = SS.XX;
        case 'covshrink',
          obj.mu =  mean(X);
          obj.Sigma =  covshrinkFit(X);
        otherwise
          error(['bad method ' method])
      end
    end
    
    function xrange = plotRange(obj, sf)
        if nargin < 2, sf = 3; end
        %if ndims(obj) ~= 2, error('can only plot in 2d'); end
        mu = mean(obj); C = cov(obj);
        s1 = sqrt(C(1,1));
        x1min = mu(1)-sf*s1;   x1max = mu(1)+sf*s1;
        if ndims(obj)==2
            s2 = sqrt(C(2,2));
            x2min = mu(2)-sf*s2; x2max = mu(2)+sf*s2;
            xrange = [x1min x1max x2min x2max];
        else
            xrange = [x1min x1max];
        end
    end
    
    

    
  end % methods


  %% Private methods
  methods(Access = 'protected')
    function checkParamsAreConst(obj)
      p = isa(obj.mu, 'double') && isa(obj.Sigma, 'double');
      if ~p
        error('params must be constant')
      end
    end
  end


end