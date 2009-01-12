classdef ConstDist < NonParamDist
  % vector of delta fns (constant values)
  properties
    point;
  end
 
  
  methods 
    function obj =  ConstDist(point)
      if nargin == 0
        point= [];
      end
      obj.point = point;
    end
  
    function d = ndimensions(obj)
      d = numel(obj.point);
    end
    
    function m = mean(obj)
      m = obj.point;
    end
    
    function m = mode(obj)
      m = obj.point;
    end  
    
    function m = var(obj)
      m = zeros(1, ndimensions(obj));
    end
    
     function point = sample(obj, n)
       % X(i,j) = sample from params(j)
       point = repmat(obj.point(:)', n, 1);
     end
    
     function p = logprob(obj, X)
       % p(i,j) = log p(x(i) | params(j))
       d = ndimensions(obj);
       x = X(:);
       n = length(x);
       p = zeros(n,d);
       point = obj.point(:)';
       for j=1:d  
         p(:,j) = (x == point(j));
       end
       p = log(p+eps);
     end
     
     function logZ = lognormconst(obj)
       logZ = 0;
     end
     
      function h = plot(obj)
            stem(obj.point(:)','LineWidth',3);
            xlabel('dimension');
            title('constant distribution');
            axis tight;
            grid on;
      end
            
  end
  
  methods(Static = true)
      
      function testClass()
         point = 10*rand(20,1);
         p = ConstDist(point);
         logprob = logProb(p,point);
         s = sample(p,10);
         logZ = lognormconst(p);
         nd = ndimensions(p);
         m = mean(p);
         v = var(p);
         mm = mode(p);
         plot(p);
      end
      
  end
    
end