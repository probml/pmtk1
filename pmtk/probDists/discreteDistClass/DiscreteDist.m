classdef DiscreteDist  < ParamDist
% This class represents a distribution over a discrete support
% (Multinoulli distribution).

  properties
    mu; %num states * num distributions
    prior;
    support;
  end
  
  %{
  properties(SetAccess = 'private')
     nstates;
     ndistrib;
  end 
 %}
  
  methods
    function obj = DiscreteDist(mu, support, prior)     
      % mu is K*d, for K states and d distributions.
      % Each *column* of mu represents a different discrete distribution. 
      % Support is a set of K numbers, defining the domain.
      % Each distribution has the same support.
      % Prior is 'none' or DirichletDist. Same prior is used for each distribution.
      if nargin == 0, mu = []; end
      obj.mu = mu;
      if nargin < 2, support = 1:size(mu,1); end
      obj.support = support;
      if nargin < 3, prior = 'none'; end
      obj.prior = prior;
    end

    function m = mean(obj)
       m = obj.mu; 
    end
    
    function v = var(obj)   
        v = obj.mu.*(1-obj.mu);
    end
  
    function L = logprob(obj,X)
      %L(i,j) = logprob(X(i,j) | mu(j))
      n = size(X,1); 
      ndistrib = size(obj.mu, 2);
      L = zeros(n,ndistrib);
      for j=1:ndistrib
        XX = canonizeLabels(X(:,j),obj.support);
        L(:,j) = log(obj.mu(XX,j)); % requires XX to be in 1..K
      end
    end

    function SS = mkSuffStat(obj, X)
      [nstates ndistrib] = size(obj.mu);
      counts = zeros(nstates, ndistrib);
      for j=1:ndistrib
        counts(:,j) = colvec(histc(X(:,j)), obj.support);
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
    %   'data'     X(i,j)  is the i'th value of variable j
    %
    %   'suffStat' This can be specified instead of 'data'.
    %              A struct with the fields 'counts' 
    % 
    %   'prior'    - {'none', 'dirichlet' } or DirichletDist
    %   'priorStrength' - magnitude of Dirichlet parameter
    %        (same value applied to each state/ distribution)
    %
    [X,SS,prior,priorStrength] = process_options(varargin,'data',[],'suffStat',[],...
      'prior',model.prior, 'priorStrength', model.priorStrength);
    if(isempty(model.support))
      model.support = unique(X(:));
    end
    if isempty(SS), SS = mkSuffStat(model, X); end
    [nstates ndistrib] = size(model.mu);
    switch class(prior)
      case 'DirichletDist'
        pseudoCounts = repmat(prior.alpha(:),1,ndistrib);
        model.mu = normalize(SS.counts + pseudoCounts, 1);
      case 'char'
        switch lower(prior)
          case 'none',
            model.mu = normalize(SS.counts);
          case 'dirichlet'
            pseudoCounts = repmat(priorStrength,nstates,ndistrib);
            model.mu = normalize(SS.counts + pseudoCounts);
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
      [nstates ndistrib] = size(obj.mu);
      x = zeros(n, ndistrib);
      for j=1:ndistrib
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
        [nstates ndistrib] = size(obj.mu);
        if ndistrib > 1
          error('cannot plot more than 1 distribution');
        end
        [plotArgs] = process_options(...
            varargin, 'plotArgs' ,{});
        if ~iscell(plotArgs), plotArgs = {plotArgs}; end
        h=bar(obj.mu, plotArgs{:});
        set(gca,'xticklabel',obj.support);
    end
    
    function y = mode(obj)
      % y(i) = arg max mu(:,i)
      y = obj.support(maxidx(obj.mu,[],1));
    end
    
  end

  methods(Static = true)
   
    
    function testClass()
      p=DiscreteDist([0.3 0.2 0.5]', [-1 0 1]);
      X=sample(p,1000);
      logp = logprob(p,[0;1;1;1;0;1;-1;0;-1]);
      nll  = negloglik(p,[0;1;1;1;0;1;-1;0;-1]);
      hist(X,[-1 0 1])    
    end
  end
  
  %% Getters and Setters
  %{
  methods
      function obj = set.mu(obj, mu)
          obj.mu = mu;
          obj.nstates = size(mu,1);
          obj.ndistrib = size(mu,2);
      end 
  end
  %}
end