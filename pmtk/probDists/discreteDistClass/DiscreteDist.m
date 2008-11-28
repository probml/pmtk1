classdef DiscreteDist  < ParamDist
% This class represents a distribution over a discrete support, e.g. a
% distribution over a single dice role. For a product of discrete distributions,
% see DiscreteProductDist. See also MultinomDist.
    
  properties
    support;            % the support of the distribution, e.g. 1:6
    mu;                 % the probabilities - e.g. [0.1 0.4 0.2 0.05 0.1 0.05]
  end

  
  methods
    function obj = DiscreteDist(mu, support)
    % Construct a new discrete distribution with the specified probabilities,
    % mu, and support. 
       
      if nargin == 0, mu = []; end
      if(size(mu,1) > 1)
          mu = mu';
      end
      if(size(mu,1) > 1)
          error('Only a single distribution supported - use DiscreteProductDist instead');
      end
      
      if nargin < 2, support = 1:length(mu); end
      if(~isempty(mu)),mu = normalize(mu);end
      obj.mu = mu;
      obj.support = support;
    end

    function m = mean(obj)
       m = obj.mu; 
    end
    
    function d = ndimensions(obj)
       d = numel(obj.mu); 
    end
    
    function v = var(obj)
        v = obj.mu.*(1-obj.mu);
    end
  
    
    function logp = logprob(obj, X)
    % logp(i) = log p(X(i))
    % size(logP) = size(X)
    %
    % example:
    % support = [-1,0,1]
    % mu      = [0.2,0.5,0.3]
    % d = DiscreteDist(mu,support);
    % lp = logprob(d,[-1 -1 0 0 1 -1])
    % lp =  -1.6094   -1.6094   -0.6931   -0.6931   -1.2040   -1.6094
    % exp(lp) =  0.2000    0.2000    0.5000    0.5000    0.3000    0.2000
    %
    % Note that for a general DiscreteDist p, exp(logprob(p,p.support)) == p.mu
    
        logp = reshape(log(obj.mu(canonizeLabels(X,obj.support))+eps),size(X));
    end
    
    function model = fit(model,varargin)
    % Fit the discrete distribution by counting.
    %
    % FORMAT:
    %             model = fit(model,'name1',val1,'name2',val2,...)
    %
    % INPUT:    
    %   'X'       The raw data, not the counts. Each entry in X is considered a
    %             data point so the dimensions are ignored. 
    %
    %   'suffStat' This can be specified instead of 'X'.
    %              A struct with the field 'counts' 
    %
    %
    %
        [X,suffStat,method,prior] = process_options(varargin,'X',[],'suffStat',[],'method',[],'prior',[]);
        
        if(isempty(X) && (isempty(suffStat) || ~isfield(suffStat,'counts')))
            error('You must specify either data or the sufficient statistics to fit.');
        end
        
        if(isempty(method))
           if(isempty(prior))
               method = 'mle';
           else
               method = 'map';
           end
        end
        
        if(isempty(model.support))
           if(isempty(suffStat) || ~isfield(suffStat,'support') || isempty(suffStat.support))
               model.support = unique(X(:));
           else
              model.support = suffStat.support; 
           end
        end
        
        if(isemtpy(suffStat) || ~isfield(suffStat,'counts'))
           suffStat.counts = histc(X(:),model.support);
        end
      
        switch lower(method)
            case 'mle'
                model.mu = normalize(suffStat.counts);
            case 'map'
                switch class(prior)
                    case 'DirichletDist'
                        model.mu = normalize(suffStat.counts + prior.alpha(:)' - 1);
                    otherwise
                        error('%s is not a supported prior',class(prior));
                end
            case 'bayesian'
                switch class(prior)
                    case 'DirichletDist'
                        model.mu = normalize(suffStat.counts + prior.alpha(:)');
                    otherwise
                        error('%s is not a supported prior',class(prior));
                end

            otherwise
                error('%s is an unsupported fit method',method);
            
        end
        
        
        
        
    end
   
    function x = sample(obj, n)
      % x(i) = an integer drawn from obj's support
      if nargin < 2, n = 1; end
      if isempty(obj.mu), obj = computeProbs(obj); end
      p = obj.mu; cdf = cumsum(p); 
      [dum, y] = histc(rand(n,1),[0 cdf]);
      %y = sum( repmat(cdf, n, 1) < repmat(rand(n,1), 1, d), 2) + 1;
      x = obj.support(y);
    end
    
     function h=plot(obj, varargin)
      % plot a probability mass function as a histogram
      % handle = plot(pmf, 'name1', val1, 'name2', val2, ...)
      % Arguments are
      % plotArgs - args to pass to the plotting routine, default {}
      %
      % eg. plot(p,  'plotArgs', 'r')
      [plotArgs] = process_options(...
        varargin, 'plotArgs' ,{});
      if ~iscell(plotArgs), plotArgs = {plotArgs}; end
      if isempty(obj.mu), obj = computemu(obj); end
      if isvector(obj.mu), obj.mu = obj.mu(:)'; end % 1 distributin
      n = size(obj.mu, 1);
      for i=1:n
        h=bar(obj.mu(i,:), plotArgs{:});
        set(gca,'xticklabel',obj.support);
      end
    end
    
    function y = mode(obj)
      % y(i) = arg max mu(i,:)
      [junk, ndx] = max(obj.mu,[],2);
      y = obj.support(ndx);
    end
    
    function obj = computeProbs(obj)
      % for child classes
        obj.mu = exp(logprob(obj, obj.support));
    end
    
  end

  
  
  
  
  methods(Static = true)
    
    function testClass()
      p=DiscreteDist([0.3 0.2 0.5], [-1 0 1]);
      X=sample(p,1000);
      hist(X,[-1 0 1])
       
    
    end
  end
  
end