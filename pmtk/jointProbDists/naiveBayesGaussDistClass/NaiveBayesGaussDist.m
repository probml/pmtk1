classdef NaiveBayesGaussDist < GenerativeClassifierDist

     
    methods
        
        function obj = NaiveBayesGaussDist(varargin)
           obj.nclasses = process_options(varargin,'nclasses',[]);
           obj.defaultFeaturePrior = GaussDist(0,inf);
        end
       
    end
   
    methods(Access = 'protected');
        
        
        
      
        
        function ccd = fitClassConditional(obj,X,y,c,prior)
            switch(class(prior))
                case 'GaussDist'
                    
                case 'NormInvGammaDist'
                    
                case 'InvGammaDist'
                    
                otherwise
                    error('%s is an unsupported prior',class(prior));
            
                
            end
            
        end
        
        function logp = logprobCCD(obj,X,c)
            
            
        end
      
        
        
    end
    
end

