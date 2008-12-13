classdef DiscreteDist  < ParamDist
% This class represents a distribution over a discrete support. For a product of 
% discrete distributions,  see DiscreteProductDist. 
%
% This is a scalar distribution - each data point is a scalar value representing
% the actual outcome of an event not the counts as with the MultinomDist class.
% Here N, (e.g. the number of dice rolls) is always one. To illustrate this
% difference, suppose we roll a die exactly once, (N=1) and see a 5. The
% DiscreteDist models this directly, i.e. the input to the fit function is the 
% data point 5 itself, whereas the input to the MultinomDist.fit method would be
% the vector of counts [0 0 0 0 1 0]. If we rolled the dice twice and recieved a
% 5 and then a 6, the DiscreteDist is only capable of modeling this as two
% seperate trials, e.g. [5;6] where as the MultinomDist input would be the
% single data point vector [0 0 0 0 1 1]. 
%
%% PARAMETER ACCESS
% obj.mu always returns a point estimate of mu
% obj.params always returns a distribution over mu, (although this may be a
% ConstDist).
    
  properties
    support;            % the support of the distribution, e.g. 1:6
  end
  
  properties 
  %     place holders for point estimate access such as obj.mu
  %     - actual parameters stored in params field    
  
     mu;                 % the probabilities - e.g. [0.1 0.4 0.2 0.05 0.1 0.05]
  end
  
 
  methods
    function obj = DiscreteDist(mu, support)
    % Construct a new discrete distribution with the specified probabilities,
    % mu, and support. 
    %
    % mu can be vector or one of 'DirichletDist', 'BetaDist', 'ConstDist'
       
      if nargin == 0, mu = []; end
      checkmu(obj,mu);   
      if nargin < 2
          if(isnumeric(mu)),
              support = 1:length(mu);
          else
              support = 1:ndimensions(mu); 
          end
      end
      
      if(isnumeric(mu))
          obj.params = ConstDist(mu);
      else
          obj.params = mu;
      end
      obj.support = support;
    end

    function m = mean(obj)
    % Point estimate of mu    
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
    %              model = fit(model,'name1',val1,'name2',val2,...)
    %
    % INPUT:    
    %   'data'     The raw data, not the counts. Each entry in X is considered a
    %              data point so the dimensions are ignored. 
    %
    %   'suffStat' This can be specified instead of 'data'.
    %              A struct with the fields 'counts' and 'support'.
    % 
    %   'method'   One of 'mle' | 'map' | 'bayesian'
    %
    %   'prior'    A prior on mu, e.g. a DirichletDist or BetaDist if
    %              size(obj.mu) == 2
    %
        [X,suffStat,method,prior] = process_options(varargin,'data',[],'suffStat',[],'method',[],'prior',[]);
        
               
        if(isempty(method))
           if(isempty(prior) && isa(model.params,'ConstDist'))
               method = 'mle';
           else
               method = 'bayesian';
           end
        end
        
        if(~isempty(prior))
            model.params = prior;
        else
           prior = model.params; 
        end
        
        if(isempty(model.support))
           if(isempty(suffStat) || ~isfield(suffStat,'support') || isempty(suffStat.support))
               model.support = unique(X(:));
           else
              model.support = suffStat.support; 
           end
        end
        
        if(~isstruct(suffStat) || ~isfield(suffStat,'counts'))
           suffStat = mkSuffStat(model,X);
        end
        
        switch lower(method)
            case 'mle'
                model.mu = normalize(suffStat.counts);
            case {'map','bayesian'}
                switch class(prior)
                    case 'DirichletDist'
                        model.params = DirichletDist(rowvec(suffStat.counts + colvec(prior.alpha)));
                        if(strcmpi(method,'map')),model.params = model.mu;end
                    case 'BetaDist'
                        if(size(model.mu) ~= 2),error('Use a DirichletDist instead of a BetaDist when size(model.mu) > 2');end
                        ab = rowvec(suffStat.counts + [prior.b;prior.a]);
                        model.params = BetaDist(ab(1),ab(2));
                        if(strcmpi(method,'map')),model.params = model.mu;end
                    case 'BernoulliDist'
                        model.params = BernoulliDist(normalize(prior.mu.*model.mu));
                    case 'DiscreteDist'
                        model.params = DiscreteDist(normalize(prior.mu.*model.mu),model.support);
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
      y = obj.support(maxidx(obj.mu,[],2));
    end
    
    function obj = computeProbs(obj)
      % for child classes
        obj.mu = exp(logprob(obj, obj.support));
    end
    
    function SS = mkSuffStat(obj,X,weights)
    % Construct sufficient statistics from X in a format that fit() will understand. 
    % Use of the weights is optional, e.g. for computing expected sufficient
    % statistics. Each element of X is considered a data point and so the
    % dimensions are ignored. The number of weights, if specified, must equal
    % the number of data points. 
       X = X(:);
       if(isempty(obj.support))
          SS.support = rowvec(unique(X)); 
       else
           SS.support = rowvec(obj.support);
       end
       [support,perm] = sort(SS.support);
       if(nargin > 2)
            if(numel(X) ~= numel(weights))
                error('The number of weights, %d, does not equal the number of data points %d',numel(weights),numel(X));
            end
            SS.counts = zeros(numel(support),1);
            weights = weights(:);
            for i=1:numel(support)
                SS.counts(i) = sum(weights(X == support(i))); %#ok
            end
            SS.counts = SS.counts(perm);
       else
            if(isempty(obj.support))
               obj.support = rowvec(unique(X)); 
            end
            counts = histc(X,support);
            SS.counts = counts(perm);
       end
    end
    
  end

  %% Getters and Setters
  methods
      
      function m = get.mu(obj)
          m = mode(obj.params);
      end
      
      function obj = set.mu(obj,val)
          if(isnumeric(val))
              val = ConstDist(val);
          end
          obj.params = val;
      end 
  end
  
  
  
  
  methods(Access = 'protected')
      
      function checkmu(obj,mu)
          switch class(mu)
              case 'double'
                  if(size(mu,1) > 1 && size(mu,2) > 1)
                     error('Use DiscreteProductDist to represent a vectorized product of DiscreteDists, here mu must be a vector'); 
                  end
              case {'ConstDist','DirichletDist','BetaDist'}
                  
              otherwise
                  error('%s is not a supported class type/prior for mu',class(mu));
          end
          
          
      end
      
      
  end
  
  
  
  methods(Static = true)
    
    function testClass()
      p=DiscreteDist([0.3 0.2 0.5], [-1 0 1]);
      X=sample(p,1000);
      logp = logprob(p,[0;1;1;1;0;1;-1;0;-1]);
      nll  = negloglik(p,[0;1;1;1;0;1;-1;0;-1]);
      hist(X,[-1 0 1])
      
       
    
    end
    
    
  end
  
end