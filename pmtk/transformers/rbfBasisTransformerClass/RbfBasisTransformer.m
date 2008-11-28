classdef RbfBasisTransformer < Transformer
  
  properties
    ncenters;
    centers;
    sigma;
  end

  %%  Main methods
  methods
    function obj = RbfBasisTransformer(ncenters, sigma)
      if nargin == 0
        ncenters = []; sigma = []; 
      end
      obj.ncenters = ncenters;
      obj.sigma = sigma;
    end
    
    function d=ndimensions(obj, X)
      d = obj.ncenters;
    end
     
    function [K, obj] = train(obj, X)
       if isvector(X)
        x = X(:);
        obj.centers = linspace(min(x), max(x), obj.ncenters)';
      else
        % dumb initialization - pick random subset
        % could use clustering
        [n d] = size(X);
        perm = randperm(n);
        obj.centers = X(perm(1:obj.ncenters),:);
      end
      K = rbfKernel(X, obj.centers, obj.sigma);
    end
    
     function [K] = test(obj, X)
      K = rbfKernel(X, obj.centers, obj.sigma);
     end
    
  end

end