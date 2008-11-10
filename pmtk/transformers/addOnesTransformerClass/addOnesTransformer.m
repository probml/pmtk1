classdef AddOnesTransformer < Transformer
  
  properties
  end

  %%  Main methods
  methods
    function obj = AddOnesTransformer()
      obj.dummy = 1;
    end
    
    function [Xnew, obj] = train(obj, X)
      % Xnew is same as X except first column is all 1s
      [n d] = size(X);
      Xnew = [ones(n,1) X];
    end
    
    function [Xnew] = test(obj, X)
      % Xnew is same as X except first column is all 1s
      [n d] = size(X);
      Xnew = [ones(n,1) X];
    end
    
    function p = addOffset(obj)
      p = true;
    end
  end

end