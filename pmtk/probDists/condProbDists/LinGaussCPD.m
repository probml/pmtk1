classdef LinGaussCPD < CondProbDist 
  % p(y|x) = N(y | w'*x + w0, v) where x is all the parents
  
  properties 
    w;
    w0;
    v; % variance
    %domain;
  end
  
  methods
    function obj = LinGaussCPD(w, w0, v)
      if(nargin == 0),w=[];w0=[];v=[];end
      % domain = indices of each parent, followed by index of child
      %obj.domain = domain;
      obj.w = w; obj.w0 = w0; obj.v = v;
    end
    
      function p = isDiscrete(CPD) %#ok
        p = false;
      end
      
      function q = nstates(CPD)  %#ok
        q = 1;
      end
      
      function Tfac = convertToTabularFactor(CPD, child, ctsParents, dParents, visible, data, nstates,fullDomain) %#ok
        error('cannot convert LinGaussCPD to tabular')
      end
      
    function obj = fit(obj, varargin)
        % X(i,:) are the values of the parents in the i'th case
        % y(i) is the value of the child
        [X, y] = process_options(varargin, ...
            'X', [], 'y', []);
        n = size(X,1);
        X1 = [ones(n,1) X];
        w = X1 \ y;
        obj.w = w(2:end);
        obj.w0 = w(1);
        obj.v = var(X1*w - y);
    end
    
    function ll = logprob(obj, Xpa, Xself)
        % ll(i) = log p(X(i,self) | X(i,pa), params)
        [n d] = size(Xpa);
        X = [ones(n,1) Xpa];
        mu = X * [obj.w0;obj.w];
        sigma2 = obj.v*ones(n,1);
        %pgauss = GaussDist(mu, sigma2);
        %ll = logprob(pgauss, Xself);
        ll = log(normpdf(Xself, mu, sqrt(sigma2)));
    end
    
    function y = sample(obj, Xpa, n)
        y = zeros(n,1);
        [n d] = size(Xpa);
        X = [ones(n,1) Xpa];
        mu = X * [obj.w0;obj.w];
        sigma = sqrt(obj.v)*ones(n,1);
        y = mu + sigma .* randn(n, 1);
    end
    
  end
  
end
