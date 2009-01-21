classdef DiscreteDist  < ParamDist
% This class represents a distribution over a discrete support
% (Multinoulli distribution).

  properties
    mu; %num states * num distributions
    prior;
    support;
  end
  
  
  methods
    function obj = DiscreteDist(varargin)  
      % 'mu' - mu is K*d, for K states and d distributions.
      % Each *column* of mu represents a different discrete distribution. 
      % 'support' - Support is a set of K numbers, defining the domain.
      % Each distribution has the same support.
      % 'prior' - 'none' or DirichletDist. Same prior is used for each distribution.
      [mu, support, prior] = process_options(varargin, ...
        'mu', [], 'support', [], 'prior', 'none');
      if isempty(support) && ~isempty(mu)
        [nstates] = size(mu,1);
        support = 1:nstates;
      end
      if(~approxeq(normalize(mu,1),mu))
         error('Each column must sum to one'); 
      end
      obj.mu = mu;
      if isempty(support) && nargin > 1
          error('must specify support'); 
      end
      obj.support = support;
      obj.prior = prior;
    end

    function d = ndistrib(obj)
      d = size(obj.mu, 2);
    end
    
    function K = nstates(obj)
      K = length(obj.support); % size(obj.mu, 1);
    end
    
    
    function p = pmf(obj)
      % p(j,d) = p(X=j | params(d)), j=1:nstates, d=1:ndistrib
      p = obj.mu;
    end
    
    function m = mean(obj)
       m = obj.mu; 
    end
    
    function v = var(obj)   
        v = obj.mu.*(1-obj.mu);
    end
  
    function L = logprob(obj,X)
      %L(i,j) = logprob(X(i) | mu(j)) or logprob(X(i,j) | mu(j)) for X(i,j)
      %in support
      n = size(X,1); 
      d = ndistrib(obj);
      if size(X,2) == 1, X = repmat(X, 1, d); end
      L = zeros(n,d);
      for j=1:d
        XX = canonizeLabels(X(:,j),obj.support);
        L(:,j) = log(obj.mu(XX,j)); % requires XX to be in 1..K
      end
    end
    
    function p = predict(obj)
      % p(j) = p(X=j)
      p = obj.mu;
    end
    
    
    function obj = mkRndParams(obj,d,ndistrib)
       if(nargin < 3)
           ndistrib = 1;
       end
       obj.mu = normalize(rand(d,ndistrib));
       obj.support = 1:d;
    end
    
    function SS = mkSuffStat(obj, X,weights)
        K = nstates(obj); d = size(X,2);
        counts = zeros(K, d);
        if(nargin < 3)
            for j=1:d
                counts(:,j) = colvec(histc(X(:,j), obj.support));
            end
        else  %weigthed SS
            if size(weights,2) == 1
                weights = repmat(weights,1,d);
            end
            for j=1:d
                for s=1:K
                    counts(s,j) = sum(weights(X(:,j) == obj.support(s),j));
                end
            end
        end
        SS.counts = counts;
    end
    
    function model = fit(model,varargin)
    % Fit the discrete distribution by counting.
    %
    % FORMAT:
    %              model = fit(model,'name1',val1,'name2',val2,...)
    %
    % INPUT:    
    %   'data'     X(i,j)  is the i'th value of variable j, in obj.support
    %
    %   'suffStat' - A struct with the fields 'counts'. Each column j is
    %   histogram over states for distribution j.
    % 
    %   'prior'    - {'none', 'dirichlet' } or DirichletDist
    %   'priorStrength' - magnitude of Dirichlet parameter
    %        (same value applied to each state/ distribution)
    %
    [X,SS,prior,priorStrength] = process_options(varargin,'data',[],'suffStat',[],...
      'prior', model.prior, 'priorStrength', 0);
    if(isempty(model.support))
      model.support = unique(X(:));
    end
    if isempty(SS), SS = mkSuffStat(model, X); end
    K = nstates(model); d = ndistrib(model);
    switch class(prior)
      case 'DirichletDist'
        pseudoCounts = repmat(prior.alpha(:),1,d);
        model.mu = normalize(SS.counts + pseudoCounts, 1);
      case 'char'
        switch lower(prior)
          case 'none'
            model.mu = normalize(SS.counts,1);
           
          case 'dirichlet'
            pseudoCounts = repmat(priorStrength,K,d);
            model.mu = normalize(SS.counts + pseudoCounts,1);
          otherwise
            error(['unknown prior %s ' prior])
        end
      otherwise
        error('unknown prior ')
    end

    end
   
    function x = sample(obj, n)
      % x(i,j) in support for i=1:n, j=1:ndistrib
      if nargin < 2, n = 1; end
      K = nstates(obj); d = ndistrib(obj);
      x = zeros(n, d);
      for j=1:d
        p = obj.mu(:,j); cdf = cumsum(p);
        [dum, y] = histc(rand(n,1),[0 ;cdf]);
        x(:,j) = obj.support(y);
      end
    end
    
    function h=plot(obj, varargin)
        % plot a probability mass function as a histogram
        % handle = plot(pmf, 'name1', val1, 'name2', val2, ...)
        % Arguments are
        % plotArgs - args to pass to the plotting routine, default {}
        %
        % eg. plot(p,  'plotArgs', 'r')
        d = ndistrib(obj);
        if d > 1, error('cannot plot more than 1 distribution'); end
        [plotArgs] = process_options(varargin, 'plotArgs' ,{});
        if ~iscell(plotArgs), plotArgs = {plotArgs}; end
        h=bar(obj.mu, plotArgs{:});
        set(gca,'xticklabel',obj.support);
    end
    
    function y = mode(obj)
      % y(i) = arg max mu(i,:)
      y = obj.support(maxidx(obj.mu,[],1));
      y = y(:);
    end
    
  end

  methods(Static = true)
    function testClass()
      p=DiscreteDist('mu', [0.3 0.2 0.5]', 'support', [-1 0 1]);
      X=sample(p,1000);
      logp = logprob(p,[0;1;1;1;0;1;-1;0;-1]);
      nll  = negloglik(p,[0;1;1;1;0;1;-1;0;-1]);
      p = fit(p, 'data', X, 'prior', 'dirichlet', 'priorStrength', 2);
    end
  end
  
 
end