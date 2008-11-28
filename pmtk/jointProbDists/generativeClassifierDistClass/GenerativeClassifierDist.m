classdef GenerativeClassifierDist < ProbDist
% This is an abstract super class for Generative Classifiers. Subclasses must 
% implement two protected methods: fitClassConditional() and logProbCCD().
    
    properties
        nclasses;                       % classes in 1:K
        classConditionalDensities;      % a cell array 
        classPosterior;                 % a Dirichlet distribution
        defaultFeaturePrior;            % default feature prior used if the user does not specify one.
                                        % This must be set by subclasses
        classSupport;                   % the support of the class labels, e.g. 1:10
        transformer;                    % a data transformer object
    end
    
    methods
        
        function obj = fit(obj,varargin)
        % Fit the classifier
        %
        % FORMAT:
        %           obj = fit(obj,'name1',val1,'name2',val2,...)
        % INPUT:
        %           'X'            - the training examples, X(i,:) is the ith case
        %
        %           'y'            - the training labels
        %
        %           'classPrior'   - a DirichletDist, if not specified, an
        %                            uninformative prior is used instead, e.g. all ones.
        %
        %           'featurePrior' - the distribution type depends on the implementing
        %                            subclass.
        % OUTPUT:
        %           obj            - the fitted model
        %           
            [X,y,classPrior,featurePrior] = process_options(varargin,...
                'X',[],'y',[],'classPrior',[],'featurePrior',[]);
            
            if(isempty(obj.nclasses))
                obj.nclasses = numel(unique(y));
            end
            
            if(isempty(classPrior))
                classPrior = DirichletDist(ones(1,obj.nclasses)); %uninformative prior
            end
            
            if(isempty(featurePrior))
               featurePrior = obj.defaultFeaturePrior; 
            end
            
            [y,classSupport] = canonizeLabels(y);
            if(isempty(obj.classSupport)),obj.classSupport = classSupport;end
            
            Nc = histc(y,1:obj.nclasses);
            obj.classPosterior = DirichletDist(Nc(:)' + classPrior.alpha);
            
            if(~isempty(obj.transformer))
               [X,obj.transformer] = train(obj.transformer,X); 
            end
            
            for c=1:obj.nclasses
                obj.classConditionalDensities{c} = fitClassConditional(obj,X,y,c,featurePrior);
            end
            
        end
        
        function pred = predict(obj,X)
        % Return a predictive distribution over class labels, (one for each 
        % training case in X, vectorized into a single DiscreteDist object).
            if(~isempty(obj.transformer))
               X = train(obj.transformer,X); 
            end
            logprobs = logprob(obj,X);
            probs = exp(logprobs);
            pred = DiscreteDist(probs,obj.classSupport);
        end
        
        function L = logprob(obj,X)
        % log probability of the data under the full posterior.    
            logpy = log(mean(obj.classPosterior));  
            L = zeros(size(X,1),obj.nclasses);
            for c=1:obj.nclasses
                L(:,c) = logprobCCD(obj,X,c) + logpy(c);
            end
            p = exp(L);
            L = log(bsxfun(@rdivide,p,sum(p,2)));   % normalize
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
    
    methods(Access = 'protected', Abstract = true)
    % These methods must be implemented by subclasses     
        
        ccd = fitClassConditional(obj,X,y,c,prior);
        % Fit and return the class conditional density for class
        
        logp = logprobCCD(obj,X,c);
        % Return the log probability of the data X under the class conditional
        % density for class c. 
    end
    
    
end

