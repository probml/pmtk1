classdef GenerativeClassifierDist < ProbDist
    
    
    properties(Abstract = true)
        nclasses;                       % classes in 1:K
        classConditionalDensities;
        classPosterior;
    end
    
    methods
        
        function obj = fit(obj,varargin)
            
            [X,y,classPrior,featurePrior] = process_options(varargin,...
                'X',[],'y',[],'classPrior',[],'featurePrior',[]);
            
            if(isempty(classPrior))
                classPrior = DirichletDist(ones(1,obj.nclasses)); %uninformative prior
            end
            
            Nc = histc(canonizeLabels(y),1:obj.nclasses);
            obj.classPosterior = DirichletDist(Nc + classPrior.alpha);
            
            for c=1:obj.nclasses
                obj.classConditionalDensities{c} = obj.fitClassConditional(obj,X(Y==c,:),c,featurePrior);
            end
            
        end
        
        function pred = predict(obj,X)
        
            logprobs = logprob(obj,X,1:obj.nclasses);
            pred = DiscreteDist(exp(logprobs));
            
        end
        
        function L = logprob(obj,X,y)
            L = 0;
            py = mean(obj.classPosterior);  
            for i=1:numel(y)
                L = L + logprob(obj.classConditionalDensities{y(i)},X)+ logprob(py(i));
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
            X = sample(obj.classConditionalDensities{y},n);
        end
        
    end
    
    methods(Access = 'protected', Abstract = true)
        obj = fitClassConditional(obj,X,y,prior);
    end
    
    
end

