classdef KernelTransformer < Transformer
  
  properties
   kernelType;
   kernelParams;
   basis;
  end

  %%  Main methods
  methods
    function obj = KernelTransformer(kernelType, kernelParams)
      if nargin == 0
        kernelType = 'rbf'; kernelParams = 1;
      end
      obj.kernelType = kernelType;
      obj.kernelParams = kernelParams;
    end
    
      
    function d=ndims(obj)
      d = size(obj.basis,1);
    end
     
    function [K, obj] = train(obj, X)
      % K(i,j) = K(X(i,:), X(j,:))
      obj.basis = X;
      K = kernelBasis(X, X, obj.kernelType, obj.kernelParams);
    end
    
    function [K] = test(obj, X)
      % K(i,j) = K(X(i,:), Xtrain(j,:))
      K = kernelBasis(X, obj.basis, obj.kernelType, obj.kernelParams);
    end
    
  end

end