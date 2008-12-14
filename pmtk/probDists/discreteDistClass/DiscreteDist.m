classdef DiscreteDist  < ParamDist
% This class represents a distribution over a discrete support
% (Multinoulli distribution).

  properties
    mu; % K*d, K = num states, d = num distributions
  end
  
  properties(SetAccess = 'private')
     ndims;
  end 
 
  methods
    function obj = DiscreteDist(mu)     
      if nargin == 0, mu = []; end
      obj.mu = mu;
    end

    function m = mean(obj)
       m = obj.mu; 
    end
    
    function v = var(obj)   
        v = obj.mu.*(1-obj.mu);
    end
  
    
    function L = logprob(obj,X)
      %L(i,j) = logprob(X(i,j) | mu(j))
      n = size(X,1); d = obj.ndims;
      L = zeros(n,d);
      for j=1:d
        %XX = canonizeLabels(X(:,j));
        XX = X(:,j);
        L(:,j) = log(obj.mu(XX,j)); % requires XX to be in 1..K
      end
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
    %   'prior'    - {'none'} or DirichletDist
    %
    [X,SS,prior] = process_options(varargin,'data',[],'suffStat',[],'prior',[]);
    if isempty(SS), SS = DiscreteDist.mkSuffStat(X); end
         
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
    
    
  end

  methods(Static = true)
    function SS = mkSuffStat(X, K)
      [n d] = size(X);
      counts = zeros(d,K);
      for j=1:d
        counts(j,:) = rowvec(histc(X(:,j)));
      end
      SS.counts = counts;
    end
    
    function testClass()
      p=DiscreteDist([0.3 0.2 0.5], [-1 0 1]);
      X=sample(p,1000);
      logp = logprob(p,[0;1;1;1;0;1;-1;0;-1]);
      nll  = negloglik(p,[0;1;1;1;0;1;-1;0;-1]);
      hist(X,[-1 0 1])    
    end
  end
  
end