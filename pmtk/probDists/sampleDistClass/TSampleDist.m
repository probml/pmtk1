classdef SampleDist < ProbDist
  % Sample based representation of a pdf
  % May represent multiple distributions in pages so long as these are all
  % of he same size, e.g. the same number of samples drawn for each and each
  % of the same dimensionality. See LogregDist.predict('method','mc') for an
  % example of this usage. 
  
  properties
    samples; % rows are samples, columns are dimensions
             % multiple pdfs may be represented in a single object where the
             % sth sample for the dth dimension of the ith pdf is stored in 
             % samples(s,d,i)
  end

  %%  Main methods
  methods
    function m = SampleDist(X)
    % Constructor    
      if nargin < 1, X = []; end
      m.samples = X;
    end
    
    function mu = mean(obj) 
    % Average samples, (if this object represents multple pdfs, do this for each
    % in parallel). mu is of size npdfs-by-d. 
      mu = squeeze(mean(obj.samples))';
    end
    
    function mu = median(obj)
    % mu is of size npdfs-by-d    
      mu = squeeze(median(obj.samples))';
    end
    
    function m = mode(obj)
    % m is of size npdfs-by-d
      [ndx, m] = max(obj.samples);
      m = squeeze(m)';
    end
    
    function v = var(obj)
    % v is of size npdfs-by-d    
      v = squeeze(var(obj.samples))';
    end
    
    function C = cov(obj)
    % C is of size d-by-d-npdfs
      [n,d,npdfs] = size(obj.samples);  
      C = zeros(d,d,npdfs);
      for i=1:npdfs
         C(:,:,i) = cov(obj.samples);
      end
    end
    
    function mm = marginal(m, queryVars)    
    % mm is of size nsamples-by-numel(queryVars)-by-npdfs    
      mm = SampleDist(m.samples(:,queryVars,:));
    end
    
    function mm = extractDist(m,ndx)
    % mm is a SampleDist object of size nsamples-by-ndimensions-by-numel(ndx)    
    % When this object represents multiple distributions, you can extract one or
    % more of them and have them returned in a new sampleDist object with this
    % method. 
        mm = SampleDist(m.samples(:,:,ndx));
    end
    
    function s = extractSample(m,sampleNDX)
    % Extract the specified sample(s)
    % s is numel(sampleNDX)-by-ndimensions-by-npdfs - a double matrix, not an
    % object. If numel(sampleNDX) = 1, the size is just ndimensions-by-npdfs
       s = squeeze(m.samples(sampleNDX,:,:));
    end
    
    function [l,u] = credibleInterval(obj, p,distNDX)
    % Obtain a credible interval for a single distribution, specified by distNDX.
    % If not specified, distNDX = 1, the first. 
      if(nargin < 3), distNDX = 1;end
      if(numel(distNDX)>1),error('distNDX must be a scalar');end
      samples = obj.samples(:,:,distNDX);
      if nargin < 2, p = 0.95; end
      q= (1-p)/2;
      [Nsamples d] = size(samples);
      l = zeros(d,1); u = zeros(d,1);
      for j=1:d
        tmp = sort(samples(:,j), 'ascend');     %#ok
        l(j) = tmp(floor((1-q)*Nsamples));
        u(j) = tmp(floor(q*Nsamples));
      end
    end
    
    function [h, hist_area] = plot(obj, varargin)
      [scaleFactor, useHisto,distNDX] = process_options(...
        varargin, 'scaleFactor', 1, 'useHisto', 1,'distNDX',1);
      samples = obj.samples(:,:,distNDX);
      switch nfeatures(obj)
        case 1,
          if useHisto
            [bin_counts, bin_locations] = hist(samples, 20);
            bin_width = bin_locations(2) - bin_locations(1);
            hist_area = (bin_width)*(sum(bin_counts));
            %counts = scaleFactor * normalize(counts);
            %counts = counts / hist_area;
            h=bar(bin_locations, bin_counts);
          else
            [f,xi] = ksdensity(samples(:,1));             %#ok
            plot(xi,f);
          end
        case 2, h=plot(samples(:,1), samples(:,2), '.'); %#ok
        otherwise, error('can only plot in 1d or 2d'); 
      end
    end
      
    function d = nfeatures(obj)
      % num dimensions (variables)
      %mu = mean(obj); d = length(mu);
      d = size(obj.samples,2);
    end
  end
  
  
end