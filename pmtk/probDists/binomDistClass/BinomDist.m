classdef BinomDist < ParamDist
 
  
  properties
    mu;
    N;
    prior;
  end
  
  %% Main methods
  methods 
    function obj =  BinomDist(N, mu, prior)
      % binomdist(N,mu) binomial distribution
      % N is mandatory; mu can be omitted or set to [] if it will be
      % estimated (using fit). N and mu can be vectors.
      if nargin == 0;
        N = 0; mu = [];
      end
      obj.mu = mu;
      obj.N = N;
      %obj.support = 0:N;
      if nargin < 3, prior = 'none'; end
      obj.prior = prior;
    end
    
    function d = ndistrib(obj)
      d = length(obj.mu);
    end
    
    function h=plot(obj, varargin)
      if ndistrib(obj) > 1, error('can only plot 1 distrib'); end
      [plotArgs] = process_options( varargin, 'plotArgs' ,{});
      if ~iscell(plotArgs), plotArgs = {plotArgs}; end
      h=bar(exp(logprob(obj,obj.support)), plotArgs{:});
      set(gca,'xticklabel',obj.support);
    end
    
    function X = sample(obj, n)
      % X(i,j) = sample from Binom(N(j), mu(j)) for i=1:n
       ndistrib = length(obj.mu);
       X = zeros(n, ndistrib);
       for j=1:ndistrib
        X(:,j) = sum( rand(n,obj.N(j)) < repmat(obj.mu(j), n, obj.N(j)), 2);
       end
     end
    
     function p = logprob(obj, X)
     % p(i,j) = log(p(X(i)|params(j)))
     % eg., logprob(binomdist(10,[0.5 0.1]), 1:10)
     X = X(:);
     %  p = log(binopdf(X, obj.N, obj.mu(1)));
      ndistrib = length(obj.mu);
      n = length(X);
      p = zeros(n, ndistrib);
      for j=1:ndistrib
         % LOG1P  Compute log(1+z) accurately.
        p(:,j) = nchoosekln(obj.N(j), X) + X.*log(obj.mu(j)) + (obj.N(j) - X).*log1p(-obj.mu(j));
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
    
      function SS = mkSuffStat(obj,X) %#ok
        % X(i,1) is number of successes, X(i,2) is number of failures
        % We require sum(X(i,:)) = N
       SS.counts = sum(X,1);
       SS.N = sum(X(:));
      end

     function obj = fit(obj, varargin)
       % m = fit(model, 'name1', val1, 'name2', val2, ...)
       % Arguments are
       % data - X(i,1) is number of successes, X(i,2) is number of failures
       % suffStat - SS.counts(j), SS.N = total amount of data
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
               obj.mu = SS.counts ./ SS.N;
             otherwise
               error(['unknown prior ' prior])
           end
         case 'BetaDist' % MAP estimate
           m = Binom_BetaDist(obj.N, prior);
           m = fit(m, 'suffStat', SS);
           obj.mu = mode(m.muDist);
         otherwise
            error('unknown prior ')
       end
     end
           
  end 
 
end