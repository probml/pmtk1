classdef chainTransformer < transformer
  
  properties
    transformers;
    tmp;
  end

  %%  Main methods
  methods
    function obj = chainTransformer(transformers)
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
    
    function d = nfeatures(obj, X)
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
  
  
  
  %% Demos
  methods(Static = true)
    
    function demo()
      T = chainTransformer({standardizeTransformer, addOnesTransformer});
      setSeed(0);
      Xtrain = rand(5,3);
      Xtest = rand(5,3);
      [Xtrain1, T]= train(T, Xtrain);
      [Xtest1] = test(T, Xtest);
      
      ntrain = size(Xtrain, 1); ntest = size(Xtest, 1);
      mu = mean(Xtrain); s = std(Xtrain);
      Xtrain2 = Xtrain - repmat(mu, ntrain, 1);
      Xtrain2 = Xtrain2 ./ repmat(s, ntrain, 1);
      Xtrain2 = [ones(ntrain,1) Xtrain2];
      assert(approxeq(Xtrain1, Xtrain2))
      
      Xtest2 = Xtest - repmat(mu, ntest, 1);
      Xtest2 = Xtest2 ./ repmat(s, ntest, 1);
      Xtest2 = [ones(ntest,1) Xtest2];
      assert(approxeq(Xtest1, Xtest2))
    end
    
  end

end