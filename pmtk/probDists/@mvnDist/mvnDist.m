classdef mvnDist < vecDist 
  % multivariate normal p(X|mu,Sigma) 
  
  properties
    mu;
    Sigma;
  end
  
  %% main methods
  methods
    function m = mvnDist(mu, Sigma)
      % mvnDist(mu, Sigma)
      % mu can be a matrix or a pdf, eg. 
      % mvnDist(mvnInvWishDist(...), [])
      if nargin == 0
        mu = []; Sigma = [];
      end
      m.mu  = mu;
      m.Sigma = Sigma;
      m.stateInfEng = mvnExactInfer;
    end

    function params = getModelParams(obj)
      params = {obj.mu, obj.Sigma};
    end
    
    function objS = convertToScalarDist(obj)
      if ndims(obj) ~= 1, error('cannot convert to scalarDst'); end
      objS = gaussDist(obj.mu, obj.Sigma);
    end
    
    function obj = mkRndParams(obj, d)
      if nargin < 2, d = ndims(obj); end
      obj.mu = randn(d,1);
      obj.Sigma = randpd(d);
    end
    
    function d = ndims(m)
      if isa(m.mu, 'double')
        d = length(m.mu);
      else
        d = ndims(m.mu);
      end
    end

    function logZ = lognormconst(obj)
      d = ndims(obj);
      logZ = (d/2)*log(2*pi) + 0.5*logdet(obj.Sigma);
    end
    
    function L = logprob(obj, X)
      % L(i) = log p(X(i,:) | params)
      mu = obj.mu(:)'; % ensure row vector
      if length(mu)==1
        X = X(:); % ensure column vector
      end
      [N d] = size(X);
      if length(mu) ~= d
        error('X should be N x d')
      end
      %if statsToolboxInstalled
      %  L1 = log(mvnpdf(X, obj.mu, obj.Sigma));
      M = repmat(mu, N, 1); % replicate the mean across rows
      if obj.Sigma==0
        L = repmat(NaN,N,1);
      else
        mahal = sum(((X-M)*inv(obj.Sigma)).*(X-M),2);
        L = -0.5*mahal - lognormconst(obj);
      end
      %assert(approxeq(L,L1))
    end
    
    %{
    function h=plotContour2d(obj, varargin)
      % Plot an ellipse representing the 95% contour of a Gaussian
      % eg figure; plotContour2d(mvnDist([0 0], [2 1; 1 1]))
      checkParamsAreConst(obj)
      if ndims(obj) ~= 2
        error('only works for 2d')
      end
      h = gaussPlot2d(obj.mu, obj.Sigma);
    end
     %}
  
    function mu = mean(m)
      checkParamsAreConst(m)
      mu = m.mu;
    end

    function mu = mode(m)
      mu = mean(m);
    end

    function C = cov(m)
      checkParamsAreConst(m)
      C = m.Sigma;
    end
  
    function v = var(obj)
      v = cov(obj);
    end
    
    
%     function samples = sample(obj,n)
%     % Sample n times from this distribution: samples is of size
%     % nsamples-by-ndimensions
%        if(nargin < 2), n = 1; end;
%        A = chol(obj.Sigma,'lower');
%        Z = randn(length(obj.mu),n);
%        samples = bsxfun(@plus,obj.mu(:), A*Z)';
%     end
    
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
          obj.Sigma =  covshrinkKPM(X);
        otherwise
          error(['bad method ' method])
      end
    end

    function obj = inferParams(obj, varargin)
      % Computer posterior over params
       % m = inferParams(model, 'name1', val1, 'name2', val2, ...)
       % Arguments are
       % data - data(i,:) = case i
       % suffStat - 
       %
      % if m.mu is of type mvnInvWishDist, and there is no missing
      % data, we compute the posterior exactly. Otherwise we call
      % m = infer(m.paramInfEng, m, data) to deal with it.
      %
       [X, SS] = process_options(...
         varargin, 'data', [], 'suffStat', []);
       hasMissingData =  any(isnan(X(:)));
       if hasMissingData
         obj = infer(obj.paramInfEng, obj, X);
         return;
       end
       if isempty(SS), SS = mvnDist.mkSuffStat(X); end
       if SS.n == 0, return; end
       done = false;
       switch class(obj.mu)
         case 'mvnDist'
           if isa(obj.Sigma, 'double')
             obj.mu = updateMean(obj.mu, SS, obj.Sigma);
             done = true;
           end
         case 'mvnInvWishartDist'
           obj.mu = updateMeanCov(obj.mu, SS);
           done = true;
         case 'double'
           if isa(obj.Sigma, 'invWishartDist')
             obj.Sigma = updateSigma(obj.Sigma, obj.mu, SS);
             done = true;
           end
       end
       if ~done
         obj = infer(obj.paramInfEng, obj, X);
       end
    end
    
    function [postmu, logevidence] = softCondition(pmu, py, A, y)
      % Bayes rule for MVNs
      Syinv = inv(py.Sigma);
      Smuinv = inv(pmu.Sigma);
      postSigma = inv(Smuinv + A'*Syinv*A);
      postmu = postSigma*(A'*Syinv*(y-py.mu) + Smuinv*pmu.mu);
      postmu = mvnDist(postmu, postSigma);
      %evidence = mvnpdf(y(:)', (A*pmu.mu + py.mu)', py.Sigma + A*pmu.Sigma*A');
      logevidence = logprob(mvnDist(A*pmu.mu + py.mu, py.Sigma + A*pmu.Sigma*A'), y(:)');
    end
    
  end % methods

  %% Demos
  methods(Static = true)
    function suffStat = mkSuffStat(X)
      % SS.n
      % SS.xbar = 1/n sum_i X(i,:)'
      % SS.XX(j,k) = 1/n sum_i XC(i,j) XC(i,k)
      n = size(X,1);
      suffStat.n = n;
      %suffStat.X = sum(X,1)'; % column vector
      suffStat.xbar = sum(X,1)'/n; % column vector
      Xc = (X-repmat(suffStat.xbar',n,1));
      suffStat.XX = (Xc'*Xc)/n;
    end
 
    function plot2dMarginalFigure()
      plotGauss2dMargCond;
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