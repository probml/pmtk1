classdef NaiveBayesBernoulliDist < GenerativeClassifierDist
    
    properties
        nclasses;                                           % classes in 1:K
        classConditionalDensities;
        classPosterior;
        defaultFeaturePrior = BetaDist(2,2); 
    end
    
    
    
    
    methods
        
        function obj = NaiveBayesBernoulliDist(varargin)
            obj.nclasses = process_options(varargin,'nclasses',[]);
            obj.classConditionalDensities = cell(obj.nclasses,1);
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
           
           load mnistAll;
           Xtrain = (reshape(mnist.train_images,28*28,[]))' ;
           Xtest  = (reshape(mnist.test_images,28*28,[]))'  ;
           m = mean([Xtrain(:);Xtest(:)]);
           Xtrain = Xtrain >= m;
           Xtest = Xtest >= m;
           ytrain = double(mnist.train_labels);
           ytest  = double(mnist.test_labels);
           clear mnistAll;
           nb = NaiveBayesBernoulliDist('nclasses',10);
           nb = fit(nb,'X',Xtrain,'y',ytrain,'featurePrior',BetaDist(2,2),'classPrior',DirichletDist([0,10000*ones(1,9)]));
           pred = predict(nb,Xtest);
           yhat = mode(pred);
           err = mean(yhat~=ytest)
           for i=0:9
             figure;
             imagesc(reshape(sample(nb,i),28,28));
           end
           placeFigures
           ccm = zeros(10,10);
           for i=1:10
               for j=1:10
                   ccm(i,j) = sum(yhat == i-1 & ytest == j-1);
               end
           end
           hintonDiagram(ccm);
           title('Class Confusion Matrix');
           xlabel('actual','FontSize',12);
           ylabel('predicted','FontSize',12);
        
           
           labels = {'0','1','2','3','4','5','6','7','8','9'};
           set(gca,'XTickLabel',labels,'YTicklabel',labels,'box','on','FontSize',12);
           
             
        end
        
    end
 
end