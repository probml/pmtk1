classdef GenerativeClassifierDist < ProbDist
    
    
    properties(Abstract = true)
        nclasses;                       % classes in 1:K
        classConditionalDensities;
        classPosterior;
        defaultFeaturePrior
    end
    
    properties
        classSupport; 
    end
    
    
    methods
        
        function obj = fit(obj,varargin)
            
            [X,y,classPrior,featurePrior] = process_options(varargin,...
                'X',[],'y',[],'classPrior',[],'featurePrior',[]);
            
            if(isempty(obj.nclasses))
                obj.nclasses = numel(unique(y));
            end
            
            if(isempty(classPrior))
                classPrior = DirichletDist(ones(1,obj.nclasses)); %uninformative prior
            end
            
            if(isempty(featurePrior))
               featurePrior = obj.defaultFeaturePrior; 
            end
            
            [y,classSupport] = canonizeLabels(y);
            if(isempty(obj.classSupport)),obj.classSupport = classSupport;end
            
            Nc = histc(y,1:obj.nclasses);
            obj.classPosterior = DirichletDist(Nc(:)' + classPrior.alpha);
            
            for c=1:obj.nclasses
                obj.classConditionalDensities{c} = fitClassConditional(obj,X,y,c,featurePrior);
            end
            
        end
        
        function pred = predict(obj,X)
            
            logprobs = logprob(obj,X);
            probs = exp(logprobs);
            probs = bsxfun(@rdivide,probs,sum(probs,2)); % normalize posterior 
            pred = DiscreteDist(probs,obj.classSupport);
            
        end
        
        function L = logprob(obj,X)
        % unnormalized    
            logpy = log(mean(obj.classPosterior));  
            L = zeros(size(X,1),obj.nclasses);
            for c=1:obj.nclasses
                L(:,c) = logprobCCD(obj,X,c) + logpy(c);
            end
          
        end
        
        function X = sample(obj,y,n)
            if(nargin < 2)
               y = argmax(classPosterior.sample());
            end
            if(~isscalar(y))
                error('y must be scalar as samples from different class conditional densities may not have the same dimensions.');
            end
            if(nargin < 3), n = 1;end
            y = find(y==obj.classSupport);
            if(isempty(y))
                error('%d is not in the support',y);
            end
            
            X = sample(obj.classConditionalDensities{y},n);
            
        end
        
    end
    
    methods(Access = 'protected', Abstract = true)
        
        ccd = fitClassConditional(obj,X,y,c,prior);
        logp = logprobCCD(obj,X,c);
      
      
    end
    
    
end

