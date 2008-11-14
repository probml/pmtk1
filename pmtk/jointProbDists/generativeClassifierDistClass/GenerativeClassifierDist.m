classdef GenerativeClassifierDist < ProbDist
  
    
    properties(Abstract = true)
        nclasses;
        classConditionalDensities;
        isvectorized;
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
            
            classes = unique(y);
            if(obj.isvectorized)
                obj = fitClassConditional(obj,X,y,featurePrior);
            else
               
                for c=1:obj.nclasses
                    obj.fitClassConditional(obj,X(Y==classes(c),:),classes(c),featurePrior);
                end
                
            end
            
            
        end
        
        
        function pred = predict(obj,varargin)
            
        end
        
        function L = logprob(obj,X,y)
            
            if(obj.isvectorized)
                error('not supported');
            else
               L = 0;
               for i=1:numel(y)
                  L = L + logprob(obj.classConditionalDensities{y(i)},X); 
               end
            end
        end
        
        function s = sample(obj,n)
            
            
        end
       
    end
    
    methods(Access = 'protected', Abstract = true)
        obj = fitClassConditional(obj,X,y,prior);
    end
    
    
end

