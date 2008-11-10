classdef testSetEvaluator < modelEvaluator


   properties
    
       fitFunction;                  % fitOutput  = @(model,fitParams{:})
       predictFunction;              % predOutput = @(fitOutput,predictParams{:})
       lossFunction;                 % score      = @(predOutput,lossParams{:});
       
       currentModel  = {};                 
       fitParams     = {};
       predictParams = {};
       lossParams    = {};
       
   end
   
   properties(GetAccess = 'public', SetAccess = 'private')
      scoreHistory;
      previousModel;
      previousFitResult;
      previousPredictResult;
   end

   methods
      
       function obj = testSetEvaluator()
       % Constructor    
          obj.scoreHistory    = struct('model',{},'score',{}); 
          obj.fitFunction     = @fit;
          obj.predictFunction = @predict;
          
       end
       
       function [score,obj] = evaluateModel(obj,model)
            if(nargin < 2), model = obj.currentModel;end
            if(~iscell(model)),model = {model};end
            obj.currentModel = model;
            fitOutput     = obj.fitFunction(obj.fitParams{:},model{:});
            predictOutput = obj.predictFunction(fitOutput,obj.predictParams{:});
            score         = obj.lossFunction(predictOutput,obj.lossParams{:});
            obj.previousModel = model;
            obj.previousFitResult = fitOutput;
            obj.previousPredictResult = predictOutput;
            i = numel(obj.scoreHistory);
            obj.scoreHistory(i+1).model = model;
            obj.scoreHistory(i+1).score = score;
       end
       
       function obj = setFitToIdentity(obj)
          obj.fit = @(varargin)varargin{:}; 
       end
       
       function obj = setPredefinedLoss(obj,name)
           switch lower(name)
               
               case 'mse'
                   obj.lossFunction = @(yhat,y)mean((reshape(y,size(yhat))-yhat).^2);
               case 'zeroone'
                   obj.lossFunction = @(yhat,y)sum(reshape(y,size(yhat)) ~= yhat);
               case 'meanzeroone'
                   obj.lossFunction = @(yhat,y)mean(reshape(y,size(yhat)) ~= yhat);  
               otherwise
                   error('Sorry, %s is not a predefined loss function. Please write your own and assign its handle to the lossFunction property of this object',name);
           end
           
       end
   end
   
    methods(Static = true)
        
        function testClass()
            
            meval = testSetEvaluator();
            meval = setPredefinedLoss(meval,'meanZeroOne');
               
            load crabs;
            T = ChainTransformer({StandardizeTransformer(false),KernelTransformer('rbf',2)});
            m = LogregDist('nclasses',2,'transformer',T);
            meval.fitParams     = {m,'X',Xtrain,'y',ytrain,'prior','l2','method','map','lambda'};
            meval.predictParams = {'X',Xtest,'method','plugin'};
            meval.lossParams    = {ytest};
            meval.predictFunction = @(varargin)mode(predict(varargin{:}));
            lambda = 0.1;
            [score,meval] = evaluateModel(meval,lambda);
            
    
        end
        
        
        
    end
   
end 
