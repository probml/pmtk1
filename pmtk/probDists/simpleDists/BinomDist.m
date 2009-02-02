classdef BinomDist < ParamDist
 
  
  properties
    mu;
    N;
    prior;
    support;
  end
  
  %% Main methods
  methods 
    function obj =  BinomDist(varargin)
      % binomdist(N,mu) binomial distribution
      % N is mandatory; mu can be omitted or set to [] if it will be
      % estimated (using fit). N and mu can be vectors.
      [obj.N, obj.mu, obj.prior] = process_options(varargin, ...
        'N', 0, 'mu', [], 'prior', 'none');
      obj.support = 0:obj.N;
    end
    
    function d = ndistrib(obj)
      d = length(obj.mu);
    end
    
    function h=plot(obj, varargin)
      if ndistrib(obj) > 1, error('can only plot 1 distrib'); end
      [plotArgs] = process_options( varargin, 'plotArgs' ,{});
      if ~iscell(plotArgs), plotArgs = {plotArgs}; end
      h=bar(exp(logprob(obj,obj.support')), plotArgs{:});
      set(gca,'xticklabel',obj.support);
    end
    
    function X = sample(obj, n)
      % X(i,j) = sample from Binom(N(j), mu(j)) for i=1:n
      d = ndistrib(obj);
       X = zeros(n, d);
       for j=1:d
        X(:,j) = sum( rand(n,obj.N(j)) < repmat(obj.mu(j), n, obj.N(j)), 2);
       end
     end
    
     function p = logprob(obj, X)
     % p(i,j) = log(p(X(i)|params(j))) or log(p(X(i,j)|params(j)))
     % for X(i,j) in 0:N(j)
     % eg., logprob(binomdist(10,[0.5 0.1]), 1:10)
     %  p = log(binopdf(X, obj.N, obj.mu(1)));
         
         d = ndistrib(obj);
         
         n = size(X,1);
         if size(X,2) == 1, X = repmat(X, 1, d); end
         p = zeros(n, d);
         for j=1:d
             % LOG1P  Compute log(1+z) accurately.
             Nj = obj.N(1);
             Xj = X(:,j);
             p(:,j) = nchoosekln(Nj, Xj) + Xj.*log(obj.mu(j)) + (Nj - Xj).*log1p(-obj.mu(j));
         end
     end
     
     function m = mean(obj)
       m = obj.N .* obj.mu;
     end
     
     function v = var(obj)
       v = obj.N .* obj.mu .* (1-obj.mu);
     end
     
     function m = mode(obj)
       m = floor((obj.N + 1) .* obj.mu);
     end
    
      function SS = mkSuffStat(obj,X) 
       SS.nsucc = sum(X,1);
       ntrials = size(X,1);
       SS.nfail = ntrials*obj.N - SS.nsucc;
      end

      function obj = fit(obj, varargin)
          % m = fit(model, 'name1', val1, 'name2', val2, ...)
          % Arguments are
          % data - X(i,1) is number of successes out of N
          % suffStat - struct with nsucc, nfail
          % 'prior' - 'none' or BetaDist [obj.prior]
          [X, SS, prior] = process_options(varargin,...
              'data'       , [],...
              'suffStat'   , [],...
              'prior'      , obj.prior);
          if isempty(SS), SS = mkSuffStat(obj,X); end
          switch class(prior)
              case 'char'
                  switch prior
                      case 'none'
                          obj.mu = SS.nsucc ./ (SS.nsucc + SS.nfail);
                      otherwise
                          error(['unknown prior ' prior])
                  end
              case 'BetaDist' % MAP estimate
                  m = Binom_BetaDist('N', obj.N, 'prior', prior);
                  m = fit(m, 'suffStat', SS);
                  obj.mu = mode(m.muDist);
                  
                  a = prior.a; b = prior.b;
                  mm = (SS.nsucc + a - 1) ./ (SS.nsucc + SS.nfail + a + b - 2);
                  assert(approxeq(mm, obj.mu))
              otherwise
                  error('unknown prior ')
          end
      end
     
    
           
  end 
 
end