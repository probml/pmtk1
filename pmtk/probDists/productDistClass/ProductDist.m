classdef ProductDist < ParamDist
% This class represents a product of independent probability distributions. For
% efficiency purposes, subclasses will restrict these distributions to be of the
% same type and vectorize many of the operations. This class and subclasses will
% override subsref so that the ith distribution can be extracted with pd(i). 
%
%  SUBCLASSES should override: marginal(), prod2cell(), mtimes()
    
    properties
        ndistributions = 0;     % The number of distributions
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
               
                for i=1:numel(distributions)
                   if(~isa(distributions{i},'ProbDist'))
                       error('A ProductDist can only represent a product of probability distributions. Component %d, of type %s, does not inherit from the ProbDist superclass.',i,class(distributions{i}));
                   end
                end
                obj.distArray = colvec(distributions);
                obj.ndistributions = numel(distributions);
            end
        end
        
        function d = ndimensions(obj)
        % d is a vector storing the dimensionality of each distribution. 
            d = zeros(obj.ndistributions,1);
            for i=1:obj.ndistributions
                d(i) = ndimensions(marginal(obj,i));
            end
        end
        
        function logp = logprob(obj,X)
        % evaluate the log probability of each row of X under each distribution.
        % logp is of size size(X,1)-by-ndistributions
            
            logp = zeros(size(X,1),obj.ndistributions);
            for i=1:obj.ndistributions
                logp(:,i) = colvec(logprob(marginal(obj,i),X));
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
        
        function c = prod2cell(obj)
        % return all of the distributions as a cell array - needs to be
        % overridden by subclasses. 
        
            c = obj.distArray;
            
        end
        
        function dist = subsref(obj,S)
            % Syntactic sugar for marginalize
            if(numel(S) > 1)
                dist = builtin('subsref',obj,S);
            else
                switch S.type
                    case {'()','{}'}
                        dist = marginal(obj,S.subs{1});
                    otherwise
                        dist = builtin('subsref',obj,S);
                end
            end
        end
        
        function prodDist = mtimes(obj1,obj2)
            if(isa(obj1,'ProductDist'))
               if(isa(obj2,'ProductDist'))
                  prodDist = ProductDist(vertcat(prod2cell(obj1),prod2cell(obj2))); 
               else
                  prodDist = ProductDist(vertcat(prod2cell(obj1),{obj2}));
               end
            else
                  prodDist = ProductDist(vertcat({obj1},prod2cell(obj2)));
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

