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
      m =  DiscreteDist(normalize(obj.alpha));
    end

     function SS = mkSuffStat(obj, X)
      K = nstates(obj); d = ndistrib(obj);
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
       obj.muDist = DirichletDist(obj.alpha + SS.counts);
     end
           
  end
  
  methods(Static = true)
    function testClass()
    end
  end
  
 
end