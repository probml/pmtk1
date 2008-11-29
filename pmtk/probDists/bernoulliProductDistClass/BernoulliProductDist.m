classdef BernoulliProductDist < ProductDist
% This class represents a product of independent Bernoulli distributions as
% might be useful for say NaiveBayes where this would act as a class conditional
% density. 
    
    properties
        support;    % support is a matrix of size ndistributions-by-2 indicating 
                    % the support of the ith distribution in the product. 
        mu;         % mu is a column vector and mu(i) is the mean of the ith 
                    % distribution in the product. 
    end
    
    methods
        
        function obj = BernoulliProductDist(mu,support)
            if(nargin == 1)
                support = repmat([0,1],size(mu,1),1);
            end
            if(nargin < 1),mu = [];support = [0,1];end
            obj.mu = mu(:);
            obj.support = reshape(support,[],2);
            obj.ndistributions = numel(mu);
        end
        
        
        function m = marginal(obj,ndx)
            if(numel(ndx) == 1)
                m = BernoulliDist(obj.mu(ndx),obj.support(ndx,:));
            else
                m = BernoulliProductDist(obj.mu(ndx),obj.support(ndx,:));
            end
            
        end
        
        function logp = logprob(obj,X)
            logp = X*log(obj.mu)' + (1-X)*log(1-obj.mu)'; 
        end
        
        function obj = fit(obj,varargin)
         % X is a binary matrix - rows are cases, columns are features. The ith 
         % Bernoulli is fit using only the data X(:,i);
            
            [X,suffStat,method,prior] = process_options(varargin,'X',[],'suffStat',[],'method','','prior',[]);
            if(isempty(method))
               if(~isempty(prior) || ~isdouble(obj.mu))
                   method = 'bayesian';
               end
            end
            
            if(~xor(isemtpy(suffStat),isempty(X)))
                error('You must specify exaclty one of suffStat or X to fit');
            end
            
            if(isempty(suffStat))
                suffStat = makeSuffStat(obj,X);  
            end
            obj.ndistributions = numel(suffStat.on);

            
            switch lower(method)
                
                case 'mle'
                    obj.mu = normalize(suffStat.on);
                case 'map'
                    if(isempty(prior) && isdouble(obj.mu))
                        error('You must specify a prior to do map estimation');
                    end
                    if(isempty(prior))
                        prior = obj.mu;
                    end
                    
                    switch class(prior)
                        case {'betaDist', 'betaProductDist'}
                            obj.mu = normalize(suffStat.on + colvec(prior.a));
                        otherwise
                            error('%s is not a supported prior',class(prior));
                    end
            
                    
                case 'bayesian'
                    if(isempty(prior) && isdouble(obj.mu))
                        error('You must specify a prior to do Bayesian inference');
                    end
                    % mu becomes a betaProductDist
                    error('not yet implemented');
                otherwise
                    error('%s is not a valid fit method',method);
                    
            end
            
            
        end
        
        function SS = makeSuffStat(obj,X,weights)
            X = double(logical(X));      % make sure X is binary
            if(nargin < 2)
                SS.on  = sum(X,1)';
                SS.off = (size(X,2)-SS.on)';
            else
                error('not yet implemented');
            end
        end
        
        function d = ndimensions(obj)
            d = 1;
        end
        
       
        
        function c = prod2cell(obj)
            c = cell(obj.ndistributions,1);
            for i=1:obj.ndistributions
               c{i} = BernoulliDist(obj.mu(i),obj.support(i,:)); 
            end
        end
        
    end
    
end

