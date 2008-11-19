classdef PcaTransformer < Transformer
    
    
    
    properties
        k;                  % the dimensionality of the principle components
        method;             % 1,2,3,4 selected automatically
        basisVectors;
        eigVectors;
        mu;                 % mean(Xtrain)
    end
    
    
    methods
        
        function obj = PcaTransformer(varargin)
            if(nargin == 1)
                obj.k = varargin{1};
            else
                [obj.k,obj.method] = process_options(varargin,'k',[],'method',[]);
            end
        end
        
        function [Xlow,obj] = train(obj,X)
            if(isempty(obj.k))
                obj.k = min(size(x));
            end
            obj.mu = mean(X);
            X = bsxfun(@minus,X,obj.mu);
            [obj.basisVectors, Xlow, obj.eigVectors] = pcaFast(X, obj.k,[],false);
        end
        
        function Xnew = test(obj,X)
            X = bsxfun(@minus,X,obj.mu);
            Xnew = X*obj.basisVectors;
        end
        
        
    end
    
    
    methods(Static = true)
       
        function testClass()
            
           warn = warning('off','MATLAB:nearlySingularMatrix');
           rand('twister',0);
           [Xtrain,Xtest,ytrain,ytest] = setupMnist(false);
           perm = randperm(60000);
           Xtrain = Xtrain(perm(1:5000),:);
           ytrain = ytrain(perm(1:5000),:);
           %% No PCA
           T = ChainTransformer({StandardizeTransformer(true),AddOnesTransformer()});
           m = LogregDist('nclasses',10,'transformer',T);
           m = fit(m,'X',Xtrain,'y',ytrain,'lambda',0.1);
           pred = predict(m,Xtest);
           yhat = mode(pred);
           errFull = mean(yhat~=ytest)
           %% PCA k = 40 << d = 784 
           T = ChainTransformer({StandardizeTransformer(true),PcaTransformer('k',40),AddOnesTransformer()});
           m = LogregDist('nclasses',10,'transformer',T);
           m = fit(m,'X',Xtrain,'y',ytrain,'lambda',0.1);
           pred = predict(m,Xtest);
           yhat = mode(pred);
           errPCA = mean(yhat~=ytest)
           warning(warn);
         
        end
        
        
    end
    
    
    
end