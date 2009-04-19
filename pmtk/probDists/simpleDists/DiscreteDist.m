classdef DiscreteDist  < ParamDist
% This class represents a distribution over a discrete support
% (Multinoulli distribution).

  properties
    T; %num states * num distributions
    prior; % DirichletDist or 'none'
    priorStrength;
    support;
  end
  
  
  methods
    function obj = DiscreteDist(varargin) 
      % obj = DiscreteDist(T, nstates, support, prior, priorStrength)
      % 'T' - T is K*d, for K states and d distributions.
      %    Each *column* of T represents a different discrete distribution. 
      % 'support' - Support is a set of K numbers, defining the domain.
      % nstates - defines support to be {1,2,...,nstates}
      % Each distribution has the same support.
      % 'prior' - 'none' or 'dirichlet' or DirichletDist.
      % Same prior is used for each distribution.
      if nargin == 0; return ; end
      [T, nstates, support, prior, obj.priorStrength] = processArgs(varargin, ...
        'T', [], 'nstates', [], 'support', [], 'prior', 'none', 'priorStrength', 0);
      if isempty(support) 
        if ~isempty(nstates)
          support = 1:nstates;
        elseif ~isempty(T)
          [nstates] = size(T,1);
          support = 1:nstates;
        end
      end
      % must be able to call the constructor with no args...
      %if isempty(support), error('must specify support or nstates or T'); end
      if(~approxeq(normalize(T,1),T))
         error('Each column must sum to one'); 
      end
      obj.T = T;
      obj.support = support;
      obj.prior = prior;
    end


    function d = ndistrib(obj)
      d = size(obj.T, 2);
    end
    
    function K = nstates(obj)
      K = length(obj.support); % size(obj.T, 1);
    end
    
    
    function p = pmf(obj)
      % p(j,d) = p(X=j | params(d)), j=1:nstates, d=1:ndistrib
      p = obj.T;
    end
    
    function m = mean(obj)
       m = obj.T; 
    end
    
    function v = var(obj)   
        v = obj.T.*(1-obj.T);
    end
  
  
    
    function [L,Lij] = logprob(obj,X)
      % L(i) = sum_j logprob(X(i,j) | params(j))
      % Lij(i,j) = logprob(X(i,j) | params(j))
        n = size(X,1);
        d = ndistrib(obj);
        if size(X,2) == 1, X = repmat(X, 1, d); end
        Lij = zeros(n,d);
        X = canonizeLabels(X,obj.support);
        for j=1:d
            Lij(:,j) = log(eps + obj.T(X(:,j),j)); 
        end
        L = sum(Lij,2);
    end
    
    function L = logprior(model)
      if strcmp(model.prior, 'none')
        L = 0;
      else
        L = logprob(model.prior, model.T(:)');
      end
    end
    
    function p = predict(obj)
      % p(j) = p(X=j)
      p = obj.T;
    end
    
    function obj = mkRndParams(obj,ndistrib)
      K = nstates(obj);
      if nargin < 2, ndistrib = max(1,size(obj.T,2)); end
      obj.T = normalize(rand(K,ndistrib),1);
    end
    
    function SS = mkSuffStat(obj, X,weights)
      K = nstates(obj);
      d = size(X,2);
      counts = zeros(K, d);
      X = double(full(X));
      %X = canonizeLabels(X, obj.support);
      %S = length(obj.support);
      if(nargin < 3)
        for j=1:d
          counts(:,j) = colvec(histc(X(:,j), obj.support));
        end
      else  %weighted SS
        if size(weights,2) == 1
          weights = repmat(weights,1,d);
        end
        for j=1:d
          for s=1:K % can't we vectorize this more?
            counts(s,j) = sum(weights(X(:,j) == obj.support(s),j));
          end
        end
      end
      SS.counts = counts;
    end
    
    function model = fit(model,varargin)
      % model = fit(model, data, suffStat)
      % data(i,j) is value of case i, variable j (an integer in model.support)
      % suffStat.counts is a K*d matrix
      [X, SS] = processArgs(varargin, ...
        'data', [], ...
        'suffStat', []);
      if isempty(SS), SS = mkSuffStat(model, X); end
      %K = nstates(model);
      d = size(SS.counts,2);
      if isempty(model.prior), model.prior = 'none'; end
      if isa(model.prior, 'char'), model = initPrior(model, X); end
      switch class(model.prior)
        case 'DirichletDist'
          pseudoCounts = repmat(model.prior.alpha(:),1,d);
          model.T = normalize(SS.counts + pseudoCounts -1, 1);
        otherwise
          error('unknown prior ')
      end % switch prior
    end % fit
   
    
     function model = initPrior(model, X) %#ok ignores X
       %[K d] = size(SS.counts);
       if ~ischar(model.prior), return; end
       K = nstates(model);
       switch lower(model.prior)
         case 'none',
           alpha = ones(K,1);
         case 'dirichlet'
           alpha = model.priorStrength*ones(K,1);
         case 'jeffreys'
           alpha = (1/K)*ones(K,1);
         otherwise
           error(['unknown prior ' model.prior])
       end
      model.prior = DirichletDist(alpha);
     end
    
    function x = sample(obj, n)
      % x(i,j) in support for i=1:n, j=1:ndistrib
      if nargin < 2, n = 1; end
      %K = nstates(obj);
      d = ndistrib(obj);
      x = zeros(n, d);
      for j=1:d
        p = obj.T(:,j); cdf = cumsum(p);
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
        h=bar(obj.T, plotArgs{:});
        set(gca,'xticklabel',obj.support);
    end
    
    function y = mode(obj)
      % y(i) = arg max T(i,:)
      y = obj.support(maxidx(obj.T,[],1));
      y = y(:);
    end
    
    %{
    function m = marginal(obj,queryvars)
       m = DiscreteDist('T',normalize(obj.T(queryvars,:),1),'support',obj.support,'prior',obj.prior); 
    end
    %}
    
   
    
  end % methods

 
  
 
end
