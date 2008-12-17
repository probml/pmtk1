classdef Discrete_DirichletDist < CompoundDist
 % p(X,theta|alpha) = Discrete(X|theta) Dir(theta|alpha) 
  
 properties
   muDist;
   support;
 end
 
  %% Main methods
  methods 
    function obj =  Discrete_DirichletDist(muDist, support)
      % muDist is of type DirichletDist
      obj.muDist = muDist;
      if nargin < 2, support = 1:ndimensions(muDist); end
      obj.support = support;
    end

    function d = ndistrib(obj)
      d = ndistrib(obj.muDist);
    end
    
     function d = nstates(obj)
      d = ndimensions(obj.muDist);
     end
    
    
    function m = marginal(obj)
      % This may not be correct...
      m =  DiscreteDist('mu', normalize(obj.muDist.alpha));
    end

     function SS = mkSuffStat(obj, X)
      K = nstates(obj); d = size(X,2); % ndistrib(obj);
      counts = zeros(K, d);
      for j=1:d
        counts(:,j) = colvec(histc(X(:,j), obj.support));
      end
      SS.counts = counts;
     end
    

     function obj = fit(obj, varargin)
       % m = fit(model, 'name1', val1, 'name2', val2, ...)
       % Arguments are
       %   'data'     X(i,j)  is the i'th value of variable j, in obj.support
       %
       %   'suffStat' - A struct with the fields 'counts'. Each column j is
       %   histogram over states for distribution j.
       [X, SS] = process_options(varargin,...
           'data'       , [],...
           'suffStat'   , []);
       if isempty(SS), SS = mkSuffStat(obj,X); end
       d = size(X,2);
       pseudoCounts = repmat(obj.muDist.alpha, 1, d);
       obj.muDist = DirichletDist(pseudoCounts + SS.counts);
     end
           
  end
  
  methods(Static = true)
    function testClass()
      prior = DirichletDist(0.1*ones(1,3));
      X = sampleDiscrete([0.1 0.3 0.6]', 5, 2);
      m = Discrete_DirichletDist(prior);
      m = fit(m, 'data', X);
      v = var(m);
    end
  end
  
 
end