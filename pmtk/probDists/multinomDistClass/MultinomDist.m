classdef MultinomDist < DiscreteDist
  
  properties
    N; 
  end
  
  
  %% Main methods
  methods 
    function obj =  MultinomDist(N,mu)
      if nargin == 0;
        N = []; mu = [];
      end
      obj.N = N;
      obj.mu = mu;
    end
    
    function plot(obj) % over-ride default
      figure; bar(obj.mu);
      title(sprintf('Mu(%d,:)', obj.N))
    end
 
    function m = mean(obj)
      m = obj.N * obj.mu;
    end
    
    function m = mode(obj)
        m = floor((obj.N+1)*obj.mu);
    end
    
    function v = var(obj)
        v = obj.N*obj.mu*(1-obj.mu);
    end
    
   
    function X = sample(obj, n)
       % X(i,:) = random vector (of length ndimensions) of ints that sums to N,
       % for i=1:n
       if nargin < 2, n = 1; end
       if statsToolboxInstalled
          X = mnrnd(obj.N, obj.mu, n);
       else
         p = repmat(obj.mu(:), 1, n);
         X = sample_hist(p, obj.N)';
       end
     end
    
     function logp = logprob(obj, X)
       % p(i) = log p(X(i,:))
     
       n = size(X,1);
       p = repmat(obj.mu,n,1);
       xlogp = sum(X .* log(p), 2);
       logp = factorialln(obj.N) - sum(factorialln(X), 2) + xlogp; 
       % debugging
       %logp2 = log(mnpdf(X,obj.mu));
       %assert(approxeq(logp, logp2))
     end


     function logZ = lognormconst(obj)
       logZ = 0;
     end
     
     function mm = marginal(m, queryVars)
      % p(Q)
     
      dims = queryVars;
      mm = MultinomDist(m.N, m.mu(dims));
     end
    
     
     function obj = fit(obj, varargin)
       % m = fit(model, 'name1', val1, 'name2', val2, ...)
       % Arguments are
       % data - data(i,:) = vector of counts for trial i
       % suffStat - SS.counts(j), SS.N = total amount of data
       % method -  'map' or 'mle' or 'bayesian'
       [X, suffStat, method,prior] = process_options(varargin,...
           'data'       , [],...
           'suffStat'   , [],...
           'method'     , 'mle',...
           'prior'      , []);
       
       if isempty(suffStat), suffStat = mkSuffStat(obj,X); end
       if(isempty(prior))
          if(~isa(obj.params,'ConstDist'))
              prior = obj.params;
          end
       end
       
       switch method
         case 'mle'
           obj.mu =  suffStat.counts / suffStat.N;
         case 'map'
             if(isempty(prior))
                 error('No prior specified, cannot do map estimation');
             end
             switch class(prior)
                 case 'DirichletDist'
                     d = ndimensions(obj);
                     obj.mu  = (suffStat.counts + prior.alpha - 1) / (suffStat.N + sum(prior.alpha) - d);
                 case 'BetaDist'
                     if(numel(obj.mu) ~= 2)
                         error('BetaDist prior only supported for binary outcomes - use a DirichletDist prior instead');
                     end
                     ab = rowvec(colvec(suffStat.counts) + [prior.a;prior.b]);
                     obj.mu = mode(BetaDist(ab(1),ab(2)));
                 otherwise
                     error(['cannot handle mu of type ' class(obj.mu)])
             end
         case 'bayesian'
             if(isempty(prior))
                 error('No prior specified, cannot do bayesian estimation'); 
             else
                 obj.params = prior;
             end
             obj = fitBayesian(obj,'data',X,'SuffStat',suffStat);
         otherwise
           error(['unknown method ' method])
       end
     end
     
     function SS = mkSuffStat(obj,X)
       SS.counts = sum(X,2);
       SS.N = sum(X(:));
     end
  end



  
  
  %% Private methods
  methods(Access = 'protected')
   
   
    
    function obj = fitBayesian(obj, varargin)
       % m = fitBayesian(model, 'name1', val1, 'name2', val2, ...)
       % Arguments are
       % data - data(i,:) = vector of counts for trial i
       % suffStat - SS.counts(j), SS.N = total amount of data
       [X, suffStat] = process_options(...
         varargin, 'data', [], 'suffStat', []);
       if isempty(suffStat), suffStat = mkSuffStat(MultinomDist(),X); end
       switch class(obj.params)
         case 'DirichletDist'
           obj.mu = DirichletDist(obj.mu.alpha + suffStat.counts);
         case 'BetaDist'
            if(numel(obj.mu) ~= 2)
                error('BetaDist is a valid prior only when numel(obj.mu) == 2');
            end
            ab = rowvec(colvec(suffStat.counts) + [obj.params.a;obj.params.b]);
            obj.mu = BetaDist(ab(1),ab(2));
         otherwise
           error(['cannot handle mu of type ' class(obj.mu)])
       end
     end

  end
  
end