classdef Discrete_DirichletDist < ProbDist
 % p(X,theta|alpha) = Discrete(X|theta) Dir(theta|alpha) 
  
 properties
   muDist; % DirichletDist (better to call it paramDist)
   support;
 end
 
  %% Main methods
  methods 
    function obj =  Discrete_DirichletDist(varargin)
      [obj.muDist, obj.support] = processArgs(varargin, ...
        '-muDist', [], '-support', []);
      if isempty(obj.support)
        obj.support = 1:nstates(obj); 
      end
    end

    
     function d = nstates(obj)
      d = nstates(obj.muDist);
     end
    
    
     %{
    function m = marginal(obj)
      % This may not be correct...
      m =  DiscreteDist('-T', normalize(obj.muDist.alpha,2)');
    end
     %}
     
      function p = logprob(obj, X)
       % p(i) = log p(X(i)) log marginal likelihood
       % just plug in posterior mean params
       T = normalize(obj.muDist.alpha,1);
       m = DiscreteDist('-T', T);
       p = logprob(m, X);
      end
     
    function p = pmf(model)
      % predictive density - just plug in posterior mean params
      T = normalize(model.muDist.alpha,1);
      p =  DiscreteDist('-T', T);
      p = p.T; % return it as a table!
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
       if isempty(SS), SS = obj.mkSuffStat(X); end
       d = size(X,2);
       pseudoCounts = repmat(obj.muDist.alpha, 1, d);
       obj.muDist = DirichletDist(pseudoCounts + SS.counts);
     end
     
      function SS = mkSuffStat(obj, X)
          K = nstates(obj); d = size(X,2); % ndistrib(obj);
          counts = zeros(K, d);
          for j=1:d
              counts(:,j) = colvec(histc(X(:,j), obj.support));
          end
          SS.counts = counts;
      end
           
  end
  
 
  
 
end