classdef GenerativeClassifierDist < ProbDist
    
    properties
        nclasses;                       % classes in 1:K
        classConditionalDensities;      % a cell array of distributions 1 per class
        classPosterior;                 % a Dirichlet distribution
        classSupport;                   % the support of the class labels, e.g. 1:10
        transformer;                    % a data transformer object, (e.g. pcaTransformer)
    end
    
    methods
        
        function obj = GenerativeClassifierDist(varargin)
            [nclasses,obj.transformer,classConditionals,classSupport] = process_options(varargin,...
                'nclasses'              ,[],...
                'transformer'           ,[],...
                'classConditionals'     ,[],...
                'classSupport'          ,[]);
            
            if(isempty(nclasses) && ~iscell(classConditionals))
                error('Please specify the number of classes');
            end
            
            if(isempty(nclasses))
                nclasses = numel(classConditionals);
            end
            obj.nclasses = nclasses;
            
            if(isempty(classSupport))
                classSupport = 1:nclasses;
            end
            obj.classSupport = classSupport; 
            
            if(iscell(classConditionals))
                if(numel(classConditionals) ~= nclasses)
                    error('The number of classes, %d, and the number of specified class conditionals, %d, do not match',nclasses,numel(classConditionals));
                end
                obj.classConditionalDensities = classConditionals;
            else
                obj.classConditionalDensities = copy(classConditionals,nclasses,1); 
            end
            
            
        end
        
        
        
        function obj = fit(obj,varargin)
        % Fit the classifier
        %
        % FORMAT:
        %           obj = fit(obj,'name1',val1,'name2',val2,...)
        % INPUT:
        %           'dataObs'            - the training examples, dataObs(i,:) is the ith case
        %
        %           'dataHid'            - the training labels
        %
        %           'classPrior'         - a DirichletDist, if not specified, an
        %                                 uninformative prior is used instead, e.g. all ones.
        %
        %           'featurePrior'       - the distribution type depends on the implementing
        %                                 subclass.
        %           
        %           'featureFitMethod'   - 'mle', 'map', 'bayesian' 
        %
        %           'fitOptions'         - a cell array of additional arguments
        %                                  to pass to the fit method of each 
        %                                  state conditional density.
        %                                  
        %
        %
        % OUTPUT:
        %           obj            - the fitted model
        %           
            [X,y,classPrior,featurePrior,featureFitMethod,fitOptions] = process_options(varargin,...
                'dataObs',[],'dataHid',[],'classPrior',[],'featurePrior',[],'featureFitMethod','map','fitOptions',{});
            
            if(isempty(obj.nclasses))
                obj.nclasses = numel(unique(y));
            end
            
            if(isempty(classPrior))
                classPrior = DirichletDist(ones(1,obj.nclasses)); %uninformative prior
            end
            
         
            [y,classSupport] = canonizeLabels(y);
            if(isempty(obj.classSupport)),obj.classSupport = classSupport;end
            
            Nc = histc(y,1:obj.nclasses);
            obj.classPosterior = DirichletDist(Nc(:)' + classPrior.alpha);
            
            if(~isempty(obj.transformer))
               [X,obj.transformer] = train(obj.transformer,X); 
            end
            
            if(isvector(X))
                X = colvec(X);
            end
            if(isempty(fitOptions))
                for c=1:obj.nclasses
                    obj.classConditionalDensities{c} = fit(obj.classConditionalDensities{c},'data',X(y==c,:),'prior',featurePrior,'method',featureFitMethod);
                end
            else
                for c=1:obj.nclasses
                    obj.classConditionalDensities{c} = fit(obj.classConditionalDensities{c},'data',X(y==c,:),fitOptions{:});
                end
            end
            
        end
        
        function pred = predict(obj,X)
        % Return a predictive distribution over class labels, (one for each 
        % training case in X, vectorized into a single DiscreteProductDist object).
            if(~isempty(obj.transformer))
               X = test(obj.transformer,X); 
            end
            logp = logprob(obj,X,false);
            maxl = maxidx(logp,[],2);
            
            
            p = normalize(exp(logp),2);
            underflow = sum(p,2) == 0;
            p(underflow,maxl(underflow)) = 1;
            pred = DiscreteProductDist(p,obj.classSupport);
        end
        
        function L = logprob(obj,X,normalize)
        % log probability of the data under the full posterior.
            if(nargin < 3), normalize = true;end
            logpy = log(mean(obj.classPosterior));  
            if(isvector(X))
                n = length(X);
            else
                n = size(X,1);
            end
            L = zeros(n,obj.nclasses);
            
            for c=1:obj.nclasses
                L(:,c) = logprob(obj.classConditionalDensities{c},X) + logpy(c);
            end
            if(normalize)
                p = exp(L);
                L = log(bsxfun(@rdivide,p,sum(p,2)));   % normalize
            end
        end
        
        function X = sample(obj,c,n)
        % Sample n times from the class conditional density for class c. If c 
        % is not specified, it too is sampled. 
            if(nargin < 2)
               c = argmax(classPosterior.sample());
            end
            if(~isscalar(c))
                error('c must be scalar as samples from different class conditional densities may not have the same dimensions.');
            end
            if(nargin < 3), n = 1;end
            c = find(c==obj.classSupport);
            if(isempty(c))
                error('%d is not in the support',c);
            end
            
            X = sample(obj.classConditionalDensities{c},n);
            
        end
        
       
        
        function d = ndimensions(obj)
            d = ndimensions(obj.classConditionalDensities{1});
        end
        
    end
    
    methods(Static = true)
        
        function testClass()
            
            [Xtrain,Xtest,ytrain,ytest] = setupMnist(true);
            model = GenerativeClassifierDist('nclasses',10,'classConditionals',BernoulliProductDist('ndistributions',28*28),'classSupport',0:9);
            model = fit(model,'dataObs',Xtrain,'dataHid',ytrain,'ClassPrior',DirichletDist(ones(1,10)));
            pred  = predict(model,Xtest);
            errorRate = mean(mode(pred) ~= ytest);
            
            
        end
        
    end
    
  
end

