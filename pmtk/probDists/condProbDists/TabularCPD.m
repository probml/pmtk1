classdef TabularCPD < CondProbDist 

  properties 
    T;
    %domain;
    sizes;
    pseudoCounts;
  end
  
  methods
    function obj = TabularCPD(T, varargin) 
      obj.sizes = mysize(T);
      sz = obj.sizes; r = sz(end); q = prod(sz(1:end-1));
      [prior] = process_options(varargin, ...
        'prior', 'none');
      obj.T = T;
      %obj.domain = domain;
      switch lower(prior)
        case 'bdeu', C = myones(sz)*1/(q*r);
        case 'laplace', C = myones(sz)*1;
        case 'none', C = 0*myones(sz);
      end
      obj.pseudoCounts = C;
    end
    
    function Tfac = convertToTabularFactor(obj, domain,visVars,visVals)
       % domain = indices of each parent, followed by index of child
      Tfac = TabularFactor(obj.T, domain);
      if(nargin == 4 && ~isempty(visVars))
          Tfac = slice(Tfac,visVars,visVals);
      end
    end
    
    function ll = logprob(obj, Xpa, Xself)
      % ll(i) = log p(X(i,self) | X(i,pa), params)
      X = [Xpa Xself];
      sz = mysize(obj.T);
      ndx = subv2ind(sz, X); % convert data pattern into array index
      ll = log(obj.T(ndx));
      ll = ll(:);
    end

    function L = logmarglik(obj, Xpa, Xself)
      % L = int_{params} sum_i log p(X(i,self) | X(i,pa), params) p(params)
      X = [Xpa Xself];
      sz = obj.sizes;
      r = sz(end); q = prod(sz(1:end-1));
      counts = compute_counts(X', obj.sizes);
      L = sum(logmarglikDirichletMultinom(reshape(counts,q,r),...
        reshape(obj.pseudoCounts,q,r)));
    end
    
    
    function obj = fit(obj, varargin)
      [X, y] = process_options(varargin, ...
        'X', [], 'y', []);
      % X(i,:) are the values of the parents in the i'th case
      % y(i) is the value of the child
      % All values must be integers from {1,2,...,K}
      % where K is the arity of the relevant variable.     
      counts = compute_counts([X y]', obj.sizes);
      obj.T = mkStochastic(counts + obj.pseudoCounts);
    end
    
    function y = sample(obj, X, n)
      y = zeros(n,1);
      sz = mysize(obj.T); r = sz(end); q = prod(sz(1:end-1));
      T = reshape(obj.T, q, r);
      for i=1:n
        y(i) = sampleDiscrete(obj.T(X(i,:),:));
      end
    end
  end
  
end
