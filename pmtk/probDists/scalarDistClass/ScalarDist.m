classdef ScalarDist < ProbDist
  
  properties
  end

 
  methods
    function m = ScalarDist(varargin)
    end
    
    function nll = negloglik(obj, X)
      % Negative log likelihood of a data set
      % nll = -sum_i log p(X(i) | params)
      % For factorized scalar dist: nll(j) = -sum_i log p(X(i) | params(j))
      nll = -sum(logprob(obj, X),1)/size(X,1);
    end
    
    function [h,p]=plot(obj, varargin)
      % plot a density function in 1d 
      % handle = plot(pdf, 'name1', val1, 'name2', val2, ...)
      % Arguments are
      % xrange  - [xmin xmax] for 1d or [xmin xmax ymin ymax] for 2d
      % useLog - true to plot log density, default false
      % plotArgs - args to pass to the plotting routine, default {}
      % npoints - number of points in each grid dimension (default 50)
      % eg. plot(p,  'useLog', true, 'plotArgs', {'ro-', 'linewidth',2})
      % scaleFactor - scales vertical axis

      [xrange, useLog, plotArgs, useContour, npoints, scaleFactor] = process_options(...
        varargin, 'xrange', plotRange(obj), 'useLog', false, ...
        'plotArgs' ,{}, 'useContour', false, 'npoints', 100, 'scaleFactor', 1);
      if ~iscell(plotArgs), plotArgs = {plotArgs}; end
      xs = linspace(xrange(1), xrange(2), npoints);
      p = logprob(obj, xs);
      if ~useLog
        p = exp(p);
      end
      p = p*scaleFactor;
      h = plot(xs, p, plotArgs{:});
    end
  
    function xrange = plotRange(obj, sf)
      if nargin < 2, sf = 2; end
      m = mean(obj); v = sqrt(var(obj));
      xrange = [m-sf*v, m+sf*v];
    end
    
  end

  

end