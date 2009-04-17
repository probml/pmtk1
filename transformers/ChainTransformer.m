classdef ChainTransformer < Transformer
  
  properties
    transformers;
  end

  %%  Main methods
  methods
    function obj = ChainTransformer(transformers)
      if nargin == 0;return;end
      obj.transformers = transformers;
    end
    
    
    function [X, obj] = train(obj, X)
      N = length(obj.transformers);
      for i=1:N
        [X, obj.transformers{i}] = train(obj.transformers{i}, X);
      end
    end
    
    function [X] = test(obj, X)
      N = length(obj.transformers);
      for i=1:N
        [X] = obj.transformers{i}.test(X);
      end
    end
    
    function d = ndimensions(obj, X)
      X = train(obj,X);
      d = size(X,2);
      % there is probably a more efficient way...
    end
    
    function p = addOffset(obj)
      p = false;
      for i=1:numel(obj.transformers)
        if(addOffset(obj.transformers{i}))
          p = true;
          return;
        end
      end
    end

  end
  
  
  
 

end