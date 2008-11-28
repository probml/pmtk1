classdef RescaleTransformer < Transformer
  
  properties
    minVal; maxVal;
    minx; rangex;
  end

  
  %%  Main methods
  methods
    function obj = RescaleTransformer(minVal, maxVal)
      if nargin == 0
        minVal = -1; maxVal = 1;
      end
      obj.minVal = minVal; obj.maxVal = maxVal;
    end
    
    function [Xnew, obj] = train(obj, X)
      [Xnew, obj.minx, obj.rangex] = rescaleData(X, obj.minVal, obj.maxVal);
    end
    
    function [Xnew] = test(obj, X)
      [Xnew] = rescaleData(X, obj.minVal, obj.maxVal, obj.minx, obj.rangex);
    end
  
    function d = ndimensions(obj, X)
      d = size(X,2);
    end   
    
  end

end