classdef SampleBasedDist < ProbDist
  % Sample based representation of a pdf
  % samples(s,i) is sample s for dimension/ distribution i
  
  properties
    samples; % rows are samples, columns are dimensions
  end

  %%  Main methods
  methods
    function m = SampleBasedDist(X)
      if nargin == 0; return; end
      m.samples = X;
    end
    
    function mu = mean(obj) 
      mu = mean(obj.samples)';
    end
    
    function mu = median(obj)
      mu = median(obj.samples)';
    end
    
    function v = var(obj)    
      v = var(obj.samples)';
    end
    
    
    function mm = marginal(m, queryVars)
      mm = SampleBasedDist(m.samples(:, queryVars));
    end
    
    
    function [l,u] = credibleInterval(obj, p)
      if nargin < 2, p = 0.95; end
      q= (1-p)/2;
      [Nsamples d] = size(obj.samples);
      l = zeros(d,1); u = zeros(d,1);
      for j=1:d
        tmp = sort(obj.samples(:,j), 'ascend');     
        u(j) = tmp(floor((1-q)*Nsamples));
        l(j) = tmp(floor(q*Nsamples));
      end
    end
    
    
    function v = cdf(obj, x)
      [Nsamples Ndims] = size(obj.samples); 
      for j=1:Ndims
        tmp = sort(obj.samples(:,j), 'ascend'); 
        ndx = find(x <= tmp);
        ndx = ndx(1);
        v(j) = ndx/Nsamples;%#ok
      end
    end
    
    
    function [h, hist_area] = plot(obj, varargin)
      [scaleFactor, useHisto,distNDX] = process_options(...
        varargin, 'scaleFactor', 1, 'useHisto', false, 'distNDX',1);
      samples = obj.samples(:,distNDX);
      if useHisto
        [bin_counts, bin_locations] = hist(samples, 20);
        bin_width = bin_locations(2) - bin_locations(1);
        hist_area = (bin_width)*(sum(bin_counts));
        %counts = scaleFactor * normalize(counts);
        %counts = counts / hist_area;
        h=bar(bin_locations, bin_counts);
      else
        [f,xi] = ksdensity(samples);             %#ok
        plot(xi,f);
        hist_area = [];
      end
    end
 
  
  end % methods
  
end

