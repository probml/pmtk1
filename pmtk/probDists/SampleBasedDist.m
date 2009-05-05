classdef SampleBasedDist < ProbDist
  % Sample based representation of a pdf
  % samples(s,i) is sample s for dimension/ distribution i
  
  properties
    samples; % rows are samples, columns are dimensions
    domain; % integer labels for the columns
  end

  %%  Main methods
  methods
    function m = SampleBasedDist(varargin)
      [X, m.domain] = processArgs(varargin, ...
        '-X', [], '-domain', []);
      if isempty(m.domain), m.domain = 1:size(X,2); end
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
      if(isempty(m.domain))
        m.domain = 1:size(m.samples,2);
      end
      Q = lookupIndices(queryVars, m.domain);
      mm = SampleBasedDist(m.samples(:,Q,:), m.domain(Q));
      %mm = SampleBasedDist(m.samples(:, queryVars));
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
    
     function p = pmf(obj, nstates)
      % p(j) = p(X=j), j=1:nstates of the joint configuration
      % We count the number of unique joint assignments
      % We assume the samples in column j are integers {1,...,nstates(j)}
      % We can optionally specify the number of states
      [Nsamples Ndims] = size(obj.samples); 
      XX = obj.samples;
      support = cell(1,Ndims);
      if nargin < 2
        for d=1:Ndims
          support{d} = unique(XX(:,d));
          nstates(d) = length(support{d});
        end
      else
        if length(nstates)==1, nstates=nstates*ones(1,Ndims); end
        for d=1:Ndims
          support{d} = 1:nstates(d);
        end
      end
      if Ndims==1
        p = normalize(hist(XX, support{1}));
        return;
      end
      ndx = subv2ind(nstates, XX);
      K = prod(nstates);
      counts = hist(ndx, 1:K);
      p = reshape(counts,nstates)/Nsamples;
     end
    
     function s = sample(obj,n)
       NN = size(obj.samples, 1);
       if n > NN, error('requesting too many samples'); end
       perm = randperm(NN);
       ndx = perm(1:n); % randi(NN,n,1);
       s = obj.samples(ndx,:);
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

