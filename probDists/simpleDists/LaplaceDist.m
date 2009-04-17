classdef LaplaceDist < ParamDist
  % Laplace distribution or double exponential
  
  properties
    mu;
    b;
  end
  
  %% Main methods
  methods 
     function m = LaplaceDist(mu, b)
      % LaplaceDist(mu, sigma) where b is the scalar parameter
       if nargin == 0
        mu = []; b = [];
      end
      m.mu  = mu(:)';
      m.b = b(:)';
     end
     
     function d = ndimensions(obj)
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
       d = ndimensions(obj);
       for j=1:d
         u = rand(n,1) - 0.5;
         b = obj.sigma(j);
         X(:,j) = obj.mu(j) - obj.b(j) * sign(u) .* log(1-2*abs(u));
       end
     end
     
     function logZ = lognormconst(obj)
       logZ = log(2*obj.b);
     end
     
     function [L,Lij] = logprob(obj, X)
       % L(i) = sum_j logprob(X(i,j) | params(j))
      % Lij(i,j) = logprob(X(i,j) | params(j))
       d = ndimensions(obj);
       n = size(X,1);
       Lij = zeros(n,d);
       logZ = lognormconst(obj);
       for j=1:d
         x = X(:,j);
         Lij(:,j) = -(abs(x-obj.mu(j))/obj.b(j)) - logZ(j);
       end
       L = sum(Lij,2);
     end
     
     function xrange = plotRange(obj, sf)
         if nargin < 2, sf = 2; end
         m = mean(obj); v = sqrt(var(obj));
         xrange = [m-sf*v, m+sf*v];
     end
      

    
  end
  
 
  
end