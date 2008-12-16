classdef CompoundDist < ParamDist 
% p(X,theta|alpha) = p(X|theta) p(theta|alpha)   
 
  
  %% main methods
  methods
    
     
   function p = logprob(obj, X)
     p = logprob(marginal(obj), X);
   end
    
     function X = sample(obj, n)
        X = sample(marginal(obj), n);
     end
     
      function m = mean(obj)
        m = mean(marginal(obj));
      end
     
      function v = var(obj)
        v = var(marginal(obj));
      end
      
      function m = mode(obj)
        m = mode(marginal(obj));
      end
      
      function pp = predict(obj)
        pp = marginal(obj);
      end
      
  end % methods

end

  