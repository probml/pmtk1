classdef ProductDist < ParamDist
% This class represents a product of independent probability distributions. For
% efficiency purposes, subclasses will restrict these distributions to be of the
% same type and vectorize many of the operations. This class and subclasses will
% override subsref so that the ith distribution can be extracted with pd(i). 
    
    properties
        ndistributions;     % The number of distributions
    end
    
    properties(GetAccess = 'protected', SetAccess = 'protected')
       distArray;  % subclasses should store the n distributions in a 'vectorized' 
                   % way using a single parameter bank, rather than as n objects
                   % storing their own parameters, as done here for generality. 
                   
    end
    
    
    methods
        
        function obj = ProductDist(distributions)
        % construct a ProductDist from a cell array of distributions.     
            if(nargin > 0)
                if(~iscell(distributions))
                    distributions = {distributions};
                end
                obj.distArray = colvec(distributions);
                obj.ndistributions = numel(distributions);
            end
        end
        
        function d = ndimensions(obj)
        % d is a vector storing the dimensionality of each distribution. 
            d = zeros(obj.ndistributions,1);
            for i=1:obj.ndistributions
                d(i) = ndimensions(obj.distArray{i});
            end
        end
        
        function logp = logprob(obj,X)
        % evaluate the log probability of each row of X under each distribution.
        % logp is of size size(X,1)-by-ndistributions
            
            logp = zeros(size(X,1),obj.ndistributions);
            for i=1:obj.ndistributions
                logp(:,i) = colvec(logprob(obj.distArray{i},X));
            end
        end
        
        function dist = marginal(obj,ndx)
        % Should be overridden in subclasses    
            if(numel(ndx) == 1)
                dist = obj.distArray{ndx};
            else
                dist = ProductDist(obj.distArray(ndx));
            end
        end
        
        function dist = subsref(obj,S)
        % Syntactic sugar for marginalize    
            if(numel(S) > 1)
                error('Syntax Error extracting ith distribution from a ProductDist');
            end
            switch S.type
                case {'()','{}'}
                    dist = marginal(obj,S.subs{1});
                otherwise
                     error('Syntax Error extracting ith distribution from a ProductDist');
            end
        end
        
        
            
        
        
    end
    
    methods(Static = true)
        
        
        function testClass()
           d = ProductDist(copy(BernoulliDist(0.5),10,1));
           logp = logprob(d,[0;0;1]);  % returns a matrix of size 3-by-10
           t3 = d(3); 
           t13 = d(1:3);
           t13 = marginal(d,1:3);
           dim = ndimensions(d);
        end
        
        
    end
    
  
    
end

