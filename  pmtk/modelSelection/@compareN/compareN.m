classdef compareN
% Compare model prediction performance as we increase the number of training 
% examples. 
    
    properties
       
        testFunctions;      % {@(Xtrain,ytrain,Xtest)f1(Xtrain,ytrain,Xtest),...@(Xtrain,ytrain,Xtest)fn(Xtrain,ytrain,Xtest)}
        names;              % {modelName1,...,modelNameN}
        Xtrain;             % All of the training X-data, 
        Ytrain;             % All of the training Y-data, can be a cell array with transformed data for each model
        Xtest;              % All of the test X-data
        Ytest;              % All of the test Y-data
        evalPoints;         % values for n at which models will be tested.
        errorCriteria;      % {'MSE' | 'MISCLASS'}
        randomize;          % true|false  - if true, shuffle the training examples before beginning. 
        verbose;            % true|false
    end
    
    
    properties
       nmodels;
       nexamplesTotal;
       nevalPoints;
       sharedY;
    end
    
    properties
        
       errors;  % errors(n,m) = errors made by model m after evalPoints(n) training examples
      
    end
    
    
    methods
        
        function obj = compareN(varargin)
            
              [ obj.testFunctions        ,...
                obj.names                ,... 
                obj.Xtrain               ,...
                obj.Ytrain               ,...
                obj.Xtest                ,...
                obj.Ytest                ,...
                obj.evalPoints           ,...
                obj.errorCriteria        ,...
                obj.randomize            ,...
                obj.verbose           ] = ...
                process_options(varargin ,...
                'testFunction'  ,{}      ,...
                'names'         ,{}      ,...
                'Xtrain'        ,[]      ,...
                'Ytrain'        ,{}      ,...
                'Xtest'         ,[]      ,...
                'Ytest'         ,{}      ,...
                'evalPoints'    ,[]      ,...
                'errorCriteria' ,'MSE'   ,...
                'randomize'     ,true    ,...
                'verbose'       ,true    );
            
            
            if(isempty(obj.evalPoints))
                obj.evalPoints = 1:5:size(obj.Xtrain,1);
            end
            
            obj.nmodels = numel(obj.testFunctions);
            obj.nexamplesTotal = size(obj.Xtrain,1);
            obj.nevalPoints = numel(obj.evalPoints);
            obj.sharedY = ~iscell(obj.Ytrain);
           
            if(obj.randomize)
               perm = randperm(obj.nexamplesTotal);
               obj.Xtrain = obj.Xtrain(perm,:);
               if(obj.sharedY)
                  obj.Ytrain = obj.Ytrain(perm,:); 
               else
                  for i=1:obj.nmodels
                     obj.Ytrain{i} = obj.Ytrain{i}(perm,:); 
                  end
               end
            end
            
            
            if(obj.verbose)
                fprintf('Call obj.run() to begin\n');
            end
     
        end
        
        
        
        function obj = run(obj)
         
            obj.errors = zeros(obj.nevalPoints,obj.nmodels);
            for p = 1:obj.nevalPoints
                
               n = obj.evalPoints(p);
                
               for m =1:obj.nmodels
               
                   f = obj.testFunctions{m};
                   if(obj.sharedY)
                       yhat = f(obj.Xtrain(1:n,:),obj.Ytrain(1:n,:),obj.Xtest);
                       ytest = obj.Ytest;
                   else
                       yhat = f(obj.Xtrain(1:n,:),obj.Ytrain{m}(1:n,:),obj.Xtest);
                       ytest = obj.Ytest{m};
                   end
                   yhat = reshape(yhat,size(ytest));
                   if(strcmpi(obj.errorCriteria,'MSE'))
                        obj.errors(p,m) = mse(yhat,ytest);
                   else
                        obj.errors(p,m) = mean(yhat ~= ytest);
                   end
                   if(obj.verbose)
                      fprintf('model: %s\t\t,ntrain: %d\t\terror: %f\n',obj.names{m},n,obj.errors(p,m));
                   end
               end
            end
            obj = plotErrors(obj);
        end
        
        
        
        function obj = plotErrors(obj)
           figure; hold on;          
           plot(obj.evalPoints,obj.errors,'LineWidth',2);
           legend(obj.names,'Location','NorthEast');
           box on;
           title('Predictive Performance W.R.T. N');
           xlabel('number of training points');
           if(strcmpi(obj.errorCriteria,'MSE'))
               ylabel('mean squared error');
           else
               ylabel('misclassification rate');
           end
        end
        
    end
    
    
    
    
    
    
    
    
    
    
    
    
end