classdef gaussDist < scalarDist
  
  properties
    mu;
    sigma2;
    paramEstMethod = 'mle';
    clampedMu = false; clampedSigma = false;
  end
  
  %% Main methods
  methods 
     function m = gaussDist(mu, sigma2)
      % gaussDist(mu, sigma2) 
      % Note that sigma2 is the variance, not the standard deviation.
      % mu and sigma2 can be vectors; in this case, the result is a MVN with a
      % diagonal covariance matrix (product of independent 1d Gaussians).
      if nargin == 0
        mu = []; sigma2 = [];
      end
      m.mu  = mu(:)';
      m.sigma2 = sigma2(:)';
     end
     
     function d = nfeatures(obj)
       d = length(obj.mu);
     end
     
     function obj = mkRndParams(obj, d)
       % Set mu(j) and sigma(j) to random values, for j=1:d.
       if nargin < 2, d = length(obj.mu); end
       obj.mu = randn(1,d);
       obj.sigma2 = rand(1,d);
     end
     
     function mu = mean(m)
       mu = m.mu(:);
     end
     
     function mu = mode(m)
       mu = mean(m);
     end
     
     function v = var(m)
       v = m.sigma2;
     end
     
     function [l,u] = credibleInterval(obj, p)
      if nargin < 2, p = 0.95; end
      alpha = 1-p;
      sigma = sqrt(var(obj));
      mu = obj.mu;
      l = norminv(alpha/2, mu, sigma);
      u = norminv(1-(alpha/2), mu, sigma);
      z=norminv(1-alpha/2);
      assert(approxeq(l, mu-z*sigma));
      assert(approxeq(u, mu+z*sigma));
    end
     
     function X = sample(m, n)
       % X(i,j) = sample from gauss(m.mu(j), m.sigma(j)) for i=1:n
       d = nfeatures(m);
       X = randn(n,d) .* repmat(sqrt(m.sigma2), n, 1) + repmat(m.mu, n, 1);
     end

     function logZ = lognormconst(obj)
       logZ = log(sqrt(2*pi*obj.sigma2));
     end
     
     function p = logprob(obj, X)
       % p(i,j) = log p(X(i) | params(j))
       d = nfeatures(obj);
       n = length(X);
       p = zeros(n,d);
       logZ = lognormconst(obj);
       for j=1:d
         p(:,j) = (-0.5/obj.sigma2(j) * (obj.mu(j) - X).^2) - logZ(j);
       end
     end

     function obj = fit(obj, varargin)
      % m = fit(model, 'name1', val1, 'name2', val2, ...)
      % Arguments are
      % data - data(i) = case i
      % method - must be one of { mle }.
      % For Bayes or MAP estimation, use mvnDist
      [X, suffStat, method] = process_options(...
        varargin, 'data', [], 'suffStat', [], 'method', obj.paramEstMethod);
      if any(isnan(X(:)))
        error('cannot handle missing data')
      end
      switch lower(method)
        case 'mle'
          if ~obj.clampedMu, obj.mu = mean(X); end
          if ~obj.clampedSigma, obj.sigma2 = var(X,1); end
        otherwise
          error(['unrecognized method ' method])
      end
    end
  end
  
  %% Demos
  methods(Static = true)
    function demoPlot()
      xs = -3:0.01:3;
      mu = 0; sigma2 = 1;
      obj = gaussDist(mu, sigma2);
      p = exp(logprob(obj,xs));
      figure; plot(xs, p);
      figure; plot(xs, normcdf(xs, mu, sqrt(sigma2)));
    end
    
    function demoHeightWeight()
      rawdata = dlmread('heightWeightData.txt'); % comma delimited file
      data.Y = rawdata(:,1); % 1=male, 2=female
      data.X = [rawdata(:,2) rawdata(:,3)]; % height, weight
      maleNdx = find(data.Y == 1);
      femaleNdx = find(data.Y == 2);
      classNdx = {maleNdx, femaleNdx};
      fnames = {'height','weight'};
      classNames = {'male', 'female'};
      figure(1);clf
      for f=1:2
        %xrange = [0.9*min(data.X(:,f)) 1.1*max(data.X(:,f)];
        if f==1, xrange = [40 90]; else xrange = [50 300]; end
        for c=1:2
          X = data.X(classNdx{c}, f);
          pgauss(f,c) = gaussDist;
          pgauss(f,c) = fit(pgauss(f,c), 'data', X, 'method', 'mle');
          subplot2(2,2,f,c);
          plot(pgauss(f,c), 'xrange', xrange);
          title(sprintf('%s, %s', fnames{f}, classNames{c}));
          hold on
          mu = pgauss(f,c).mu;
          pmu = exp(logprob(pgauss(f,c), mu));
          line([mu mu], [0 pmu], 'color','r', 'linewidth', 2);
        end
      end
      drawnow
    end
    
    function demoCV()
      % estimate mu/sigma by cross validation over a small grid
      mu = 0; sigma = 1;
      mtrue = gaussDist(mu, sigma^2);
      ntrain = 100;
      Xtrain = sample(mtrue, ntrain);
      mus = [-10 0 10];
      sigmas = [1 1 1];
      for i=1:length(sigmas)
        models{i} = gaussDist(mus(i), sigmas(i)^2);
        models{i}.clampedMu = true;
        models{i}.clampedSigma = true;
      end
      [mestCV, cvMean, cvStdErr] = exhaustiveSearch(models, @(m) cvScore(m, Xtrain))
      mestMLE = fit(mtrue, 'data', Xtrain)
    end
    
  end
  
end