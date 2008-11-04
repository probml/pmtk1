classdef laplaceDist < scalarDist
  % Laplace distribution or double exponential
  
  properties
    mu;
    b;
  end
  
  %% Main methods
  methods 
     function m = laplaceDist(mu, b)
      % laplaceDist(mu, sigma) where b is the scalar parameter
       if nargin == 0
        mu = []; b = [];
      end
      m.mu  = mu(:)';
      m.b = b(:)';
     end
     
     function d = nfeatures(obj)
       d = length(obj.b);
     end
   
     function mu = mean(m)
       mu = m.mu;
     end
     
     function mu = mode(m)
       mu = mean(m);
     end
     
     function v = var(m)
       v = 2*(m.b .^2);
     end
     

     function X = sample(obj,n)
       % X(i,j) = sample from params(j)
        % See http://en.wikipedia.org/wiki/Laplace_distribution
       d = ndims(obj);
       for j=1:d
         u = rand(n,1) - 0.5;
         b = obj.sigma(j);
         X(:,j) = obj.mu(j) - obj.b(j) * sign(u) .* log(1-2*abs(u));
       end
     end
     
     function logZ = lognormconst(obj)
       logZ = log(2*obj.b);
     end
     
     function p = logprob(obj, X)
       % p(i,j) = log p(X(i) | params(j))
       d = nfeatures(obj);
       n = length(X);
       p = zeros(n,d);
       logZ = lognormconst(obj);
       for j=1:d
         p(:,j) = -(abs(X-obj.mu(j))/obj.b(j)) - logZ(j);
       end
     end
      

    
  end
  
 
  
end