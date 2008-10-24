classdef gaussMixDist < vecDist 
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
    function m = gaussMixDist(varargin)
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
        m = mvnDist(obj.mu(:,k), obj.Sigma(:,:,k));
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
      if isempty(SS), SS = mvnDist.mkSuffStat(X); end
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

    
  end % methods

  %% Demos
  methods(Static = true)
  
    function demoPlot()
      m = gaussMixDist;
      m.K = 3;
      m.mixweights = [0.5 0.3 0.2];
      m.mu(:,1) = [0.22 0.45]';
      m.mu(:,2) = [0.5 0.5]';
      m.mu(:,3) = [0.77 0.55]';
      m.Sigma(:,:,1) = [0.018  0.01 ;  0.01 0.011];
      m.Sigma(:,:,2) = [0.011 -0.01 ; -0.01 0.018];
      m.Sigma(:,:,3) = m.Sigma(:,:,1);
      xr = plotRange(m, 1);
       
      figure; hold on;
      colors = {'r', 'g', 'b'};
      for k=1:3
        mk = mvnDist(m.mu(:,k), m.Sigma(:,:,k));
        [h,p]=plot(mk, 'useContour', true, 'xrange', xr, 'npoints', 200);
        set(h, 'color', colors{k});
      end
      
      figure;
      h=plot(m, 'useLog', false, 'useContour', true, 'npoints', 200, 'xrange', xr);
      
      figure;
      h=plot(m, 'useLog', false, 'useContour', false, 'npoints', 200, 'xrange', xr);
      brown = [0.8 0.4 0.2];
      set(h,'FaceColor',brown,'EdgeColor','none');
      hold on;
      view([-27.5 30]);
      camlight right;
      lighting phong;
      axis off;
      
      X = sample(m, 1000);
      figure;
      h=plot(m, 'useLog', false, 'useContour', true, 'npoints', 200, 'xrange', xr);
      hold on
      plot(X(:,1), X(:,2), '.');
      axis(xr)
    end

  end


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