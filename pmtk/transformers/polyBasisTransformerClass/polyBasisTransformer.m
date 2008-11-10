classdef PolyBasisTransformer < Transformer
  
  properties
    degree;
  end

  %%  Main methods
  methods
    function obj = PolyBasisTransformer(degree)
      if nargin == 0, degree = []; end
      obj.degree = degree;
    end
    
    function [Xnew, obj] = train(obj, X)
      % Xnew(i,:) = [ 1, X(i,1), X(i,1)^2, .., X(i,1)^d,    X(i,2), X(i,2)^2, ...X(i,2)^d, ...]
      addOnes = true;
      Xnew = degexpand(X, obj.degree, addOnes);
    end
    
     function [Xnew] = test(obj, X)
       addOnes = true;
      Xnew = degexpand(X, obj.degree, addOnes);
     end
    
     function d = nfeatures(obj, X)
      d = obj.degree+1;
     end
     
     function p = addOffset(obj)
      p = true;
    end
  end

end