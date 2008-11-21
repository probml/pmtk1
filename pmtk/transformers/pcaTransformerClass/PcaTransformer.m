classdef PcaTransformer < Transformer
    
   
    properties
        k;                  % the dimensionality of the principle components
        basisVectors;
        evals;
        mu;                 % mean(Xtrain)
        method = 'default'; % one of {'default', 'eigCov', 'eigGram', 'svd'}
    end
    
    
    methods
        
        function obj = PcaTransformer(varargin)
            if(nargin == 1)
                obj.k = varargin{1};
            else
                [obj.k, obj.method] = process_options(varargin,'k',[],'method','default');
            end
        end
        
        function [Xlow,obj] = train(obj,X)
            if(isempty(obj.k))
                obj.k = min(size(x));
            end
            obj.mu = mean(X);
            X = bsxfun(@minus,X,obj.mu);
            [obj.basisVectors, Xlow, obj.evals] = pcaPmtk(X, obj.k, obj.method, false);
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
           
           if(0)
           [Xtrain,Xtest,ytrain,ytest] = setupMnist(false);
           X = [Xtrain;Xtest]; clearvars -except X ytrain ytest
           X = train(PcaTransformer('k',40),X);
           Xtrain = X(1:60000,:); 
           Xtest = X(60001:end,:);
           clear X
           T = ChainTransformer({StandardizeTransformer(true),AddOnesTransformer()});
           m = LogregDist('nclasses',10,'transformer',T);
           m = fit(m,'X',Xtrain,'y',ytrain,'lambda',0.1);
           pred = predict(m,Xtest);
           yhat = mode(pred);
           err = mean(yhat~=ytest)
           end
           
           
         
        end
        
        
    end
    
    
    
end