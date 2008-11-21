classdef NaiveBayesBernoulliDist < GenerativeClassifierDist
% Naive Bayes with products of Bernoulli distributions as class conditional
% densities. 
    
    methods
        
        function obj = NaiveBayesBernoulliDist(varargin)
        % Constructor
        %
        % FORMAT:
        %           obj = NaiveBayesBernoulliDist('name1',val1,'name2',val2,...)
        %
        % INPUT:
        %           'nclasses'     - the number of classes
        %           'transformer'  - a data transformer object, (optional)
        %
        % OUTPUT:
        %
        %          obj             - the constructed model
            [obj.nclasses,obj.transformer] = process_options(varargin,'nclasses',[],'transformer',[]);
            obj.classConditionalDensities = cell(obj.nclasses,1);
            obj.defaultFeaturePrior = BetaDist(2,2);
        end
      
    end
    
    
    methods(Access = 'protected')
        
        function ccd = fitClassConditional(obj,X,y,c,prior)
        % super class fit calls this method for every class. 
            switch(class(prior))
                case 'BetaDist'
                    Xc = X(y==c,:);
                    N1 = sum(Xc);
                    N0 = sum(1-Xc);
                    alphaN = prior.a + N1;
                    betaN =  prior.b + N0;
                    ccd = BetaDist(alphaN,betaN);
                otherwise
                    error('%s is an unsupported feature prior',class(prior));
            end 
        end
        
        function logp = logprobCCD(obj,X,c)    
            dist = obj.classConditionalDensities{c};
            m = mean(dist);                           
            logp = X*log(m)' + (1-X)*log(1-m)';
        end
        
      
        
    end
    
    methods(Static = true)
       
        function testClass()
           load soy;
           nb = NaiveBayesBernoulliDist('nclasses',3);
           nb = fit(nb,'X',X,'y',Y);
           pred = predict(nb,X);
        end
        
    end
 
end