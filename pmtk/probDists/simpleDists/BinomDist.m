classdef BinomDist < ProbDist
 
  
  properties
    mu;
    N;
    prior;
    support;
    productDist;
  end
  
  %% Main methods
  methods 
    function obj =  BinomDist(varargin)
      % binomdist(N,mu) binomial distribution
      % N is mandatory; mu can be omitted or set to [] if it will be
      % estimated (using fit). N and mu can be vectors.
      [obj.N, obj.mu, obj.prior, obj.productDist] = processArgs(varargin, ...
        '-N', 0, '-mu', [], '-prior', NoPrior, '-productDist', false);
      obj.support = 0:obj.N;
    end
    
    function T = pmf(model)
      T = model.mu;
    end
    
    function d = ndistrib(obj)
      d = length(obj.mu);
    end
    
    function h=plot(obj, varargin)
      if ndistrib(obj) > 1, error('can only plot 1 distrib'); end
      [plotArgs] = processArgs( varargin, '-plotArgs' ,{});
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
    
     function [L,Lij] = logprob(obj, X)
        % Return col vector of log probabilities for each row of X
       % L(i) = log p(X(i) | params)
       % L(i) = log p(X(i) | params(i)) (set distrib)
       % L(i) = sum_j log p(X(i,j) | params(j)) (prod distrib)
       % L = sum_j log p(X(1,j) | params(j)) (prod distrib)
       % for X(i,j) in 0:N(j)
       n = size(X,1);
       if ~obj.productDist
         X = X(:);
         if isscalar(obj.mu)
           M = repmat(obj.mu, n, 1); N = repmat(obj.N, n, 1);
         else
           M = obj.mu(:);
           N = repmat(obj.N(1), n, 1);
           %N = obj.N(:);
         end
         L = nchoosekln(N, X) + X.*log(M) + (N - X).*log1p(-M);
       else
         d = length(obj.mu);
         Lij = zeros(n, d);
         for j=1:d
           % LOG1P  Compute log(1+z) accurately.
           Nj = obj.N(1); %obj.N(j);
           Xj = X(:,j);
           Lij(:,j) = nchoosekln(Nj, Xj) + Xj.*log(obj.mu(j)) + (Nj - Xj).*log1p(-obj.mu(j));
         end
         L = sum(Lij,2);
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
        % m = fit(m, X, SS)
        [X, SS] = processArgs(varargin, ...
          '-data', [], '-SS', []);
        if isempty(SS), SS = mkSuffStat(obj,X); end
        switch class(obj.prior)
          case 'NoPrior'
            obj.mu = SS.nsucc ./ (SS.nsucc + SS.nfail);
          case 'BetaDist'
            a = obj.prior.a; b = obj.prior.b;
            obj.mu = (SS.nsucc + a - 1) ./ (SS.nsucc + SS.nfail + a + b - 2);
            if 1 % debug
              m = BinomConjugate('-N', obj.N, '-prior', obj.prior);
              m = fit(m, '-SS', SS);
              mm = mode(m.muDist);
              assert(approxeq(mm, obj.mu))
            end
          otherwise
            error('unknown prior ')
        end
      end
      
      
           
  end 
 
end