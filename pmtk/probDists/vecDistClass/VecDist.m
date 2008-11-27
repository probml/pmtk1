classdef VecDist < ProbDist
  % probability density function on vector-valued rv's (joint distributions)

  
  properties
  end

  
  %%  Main methods
  methods
    function m = VecDist(varargin)
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

  end
  
 


end