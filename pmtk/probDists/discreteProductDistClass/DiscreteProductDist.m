classdef DiscreteProductDist  < ProductDist
% A product of independent Discrete Distributions
% Each row of mu represents a single discrete distribution. 
    
    properties
       support;
       mu;          %mu(i,j) = p(x=support(j) | distribution(i))          
    end
    
    methods
            
        
        function obj = DiscreteProductDist(mu,support)
            
            if(nargin < 1), mu     = [];end
            if(nargin < 2),support = 1:numel(mu,2);end
            obj.mu = mu;
            obj.support = support;     
            
            obj.ndistributions = size(obj.mu,1);
            
        end
        
        
        function obj = fit(obj,varargin)
            [X,suffStat,method,prior] = process_options(varargin,'data',[],'suffStat',[],'method','map','prior',[]);
            
            if(~xor(isempty(X),isempty(suffStat)))
                error('You must specify exactly one of data, or suffStat');
            end
            if(isempty(suffStat))
                suffStat = mkSuffStat(obj,X);
            end
            
            if(isempty(prior))
                method = 'mle';
            end
            
            
            switch lower(method)
                case 'mle'
                    obj.mu = normalize(suffStat.counts,2);
                case {'map','bayesian'}
                    switch class(prior)
                        case {'DirichletDist','BetaDist'}
                            if(strcmpi(class(prior),'DirichletDist'))
                                alpha = prior.alpha;
                            else
                                alpha = [prior.a,prior.b];
                            end
                            if(strcmpi(method,'map'));alpha = alpha -1;end
                            alpha(alpha < 0) = 0;
                            obj.mu = normalize(bsxfun(@plus,suffStat.counts,rowvec(alpha)));
                        otherwise
                            error('%s is not a supported prior',class(prior));
                        
                    end
                  
                otherwise
                    error('unrecognized fit method');
            end
            
        end
        
        function SS = mkSuffStat(obj,X)
            if(isempty(obj.support))
                support = unique(X);
            else
                support = obj.support;
            end
            [support,perm] = sort(support);
            counts = zeros(obj.ndistributions,numel(support));
            for j=1:obj.ndistributions
               counts(j,:) = rowvec(histc(X(:,j),support)); 
            end
            SS.counts = counts(:,perm);
            SS.support = support;
        end
        
        function L = logprob(obj,X)
        %L(i,j) = logprob(X(i,j) | distribution(j)) 
            L = zeros(size(X,1),obj.ndistributions);
            for j=1:obj.ndistributions
               L(:,j) = colvec(logprob(marginal(obj,j),X(:,j)));
            end
        end
        
        function d = ndimensions(obj)
            d = size(obj.mu,2);
        end
        
        function m = marginal(obj,ndx)
            if(numel(ndx) == 1)
                m = DiscreteDist(obj.mu(ndx,:),obj.support);
            else
                m = DiscreteProductDist(obj.mu(ndx,:),obj.support);
            end
        end
        
        function s = sample(obj,n)
        %s(i,j) = sample i from distribution j 
            s = zeros(n,obj.ndistributions);
            for j=1:obj.ndistributions
               s(:,j) = colvec(sample(marginal(obj,j),n));
            end
        end
        
        function m = mean(obj)
            m = obj.mu;
        end
         
        function v = var(obj)
            v = obj.mu.*(1-obj.mu);
        end
        
        function m = mtimes(obj1,obj2) 
           error('not yet implemented'); 
        end
        
        
        function m = mode(obj)
            m = obj.support(maxidx(obj.mu,[],2))';
            
        end
        
    end
    
    
    
    methods(Access = 'protected')
        function obj = setDist(obj,ndx,newDist)
            if(~isa(newDist,'DiscreteDist'))
                error('You are trying to assign a non-DiscreteDist to a DiscreteProductDist.');
            end
            
            if(~isequal(rowvec(newDist.support),rowvec(obj.support)))
                error('support mismatch');
            end
            obj.mu(ndx,:) = rowvec(newDist.mu);
        end
        
    end
    
end