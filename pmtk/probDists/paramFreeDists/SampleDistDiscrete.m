classdef SampleDistDiscrete < SampleDist
 % Similar to SampleDist except each row of samples is interpreted as sampled
 % probabilities over a discrete support rather than a single d-dimensional
 % sample. 
    
    properties
        
    end
    
    methods
        function m = SampleDistDiscrete(X,support)
            if nargin == 0; return; end
        % Constructor    
            if nargin < 1, X = []; end
            if(nargin < 2), support = 1:size(X,2);end
            m.samples = X;
            m.support = support;
        end
        
        function m = mode(obj)
        % m is of size d-by-npdfs
            if(ndims(obj.samples) == 3)
                [val,m] = max(mean(obj),[],2);
            else
                [val, m] = max(obj.samples);
                m = squeeze(m);
            end
            m = obj.support(m);
        end

      function pred = predict(latentdist)
        % latent is a N x n matrix, where
        %   N is the number of samples
        %   n is the number of datapoints
        % Returns a length(support) x n matrix of probabilities based
        % on the sampled indicators
        K = max(latentdist.support);
        counts = histc(latentdist.samples, latentdist.support);
        pred = DiscreteDist('-T',bsxfun(@rdivide, counts, sum(counts)));
      end
        
        
        
    end
    
end

