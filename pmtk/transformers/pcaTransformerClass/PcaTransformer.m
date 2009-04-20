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
          % obj = PcaTransformer(k, method)
          [obj.k, obj.method] = processArgs(varargin, 'k', [], 'method', 'default');
          %{
            if(nargin == 1)
                obj.k = varargin{1};
            else
                [obj.k, obj.method] = process_options(varargin,'k',[],'method','default');
            end
          %}
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
    
    
  
    
end