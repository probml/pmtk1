classdef LogregDist < CondProbDist 
%% Logistic Regression, Multiclass Conditional Distribution

    properties
        w;                      
        transformer;            % A data transformer object, e.g. KernelTransformer
        nclasses;               % The number of classes
        classSupport;           % The suppport of the target y, e.g. [0,1], [-1,+1], 1:K, etc. 
        priorStrength;
    end

    
    
    
    %% Main methods
    methods

        function m = LogregDist(varargin)
        % Constructor
            [m.transformer,  m.w, m.nclasses, m.prior, m.priorStrength] = ...
              process_options( varargin ,...
                'transformer', []             ,...
                'w'          , []             , ...
                'nclasses'   , [], ...
                'prior', 'none', ...
                'priorStrength', []);
        end

        function [obj, output] = fit(obj, varargin)
        % Compute the posterior distribution over w, the weights. This is either
        % a constant distribution representing the MAP estimate if method =
        % 'map', (the default), or a full MvnDist distribution representing
        % the laplace approximation to the posterior, if method = 'bayesian'.
        %
        % FORMAT:
        %           model = fit(model, 'name1', val1, 'name2', val2, ...)
        % INPUT:
        %
        % 'X'      - The training examples: X(i,:) is the ith case
        % 'y'      - The class labels for X in {1...C}
        % 'prior'  - {'L1' | 'L2' | ['none']}]
        % 'priorStrength' - [0] regularization value
        % 'optMethod' - for L2, a method supported by minFunc
        %             - for L1, a method supported by L1general
            
            [X, y,  prior, lambda, optMethod] = process_options(varargin,...
                'X'            , []                 ,...
                'y'            , []                 ,...
                'prior'        , 'none'             ,...
                'priorStrength'       , 0                  ,...
                'optMethod'    , 'default'           );
            output = [];
            offsetAdded = false;
            if ~isempty(obj.transformer)
                [X, obj.transformer] = train(obj.transformer, X);
                offsetAdded = obj.transformer.addOffset();
            end
            if isempty(obj.nclasses), obj.nclasses = length(unique(y)); end
            obj.ndimsX = size(X,2);
            obj.ndimsY = size(y,2);
            [Y1,obj.classSupport] = oneOfK(y, obj.nclasses);
            [n,d] = size(X);   
            winit = zeros(d*(obj.nclasses-1),1);
            switch lower(prior)
              case {'l1'}
                lambdaVec = lambda*ones(d,obj.nclasses-1);
                if(offsetAdded),lambdaVec(:,1) = 0;end
                lambdaVec = lambdaVec(:);
                objective = @(w) multinomLogregNLLGradHessL2(w, X, Y1,0,false);
                options.verbose = false;
                if(strcmpi(optMethod,'default'))
                  optMethod = 'L1GeneralProjection';
                end
                [w,fEvals] = L1General(optMethod, objective, winit,lambdaVec, options);
              case {'l2', 'none'}
                objective = @(w) multinomLogregNLLGradHessL2(w, X, Y1,lambda,offsetAdded);
                if(strcmpi(optMethod,'default'))
                  optMethod = 'lbfgs';
                end
                options.Method = optMethod;
                options.Display = false;
                [w, f, exitflag, output] = minFunc(objective, winit, options);
              otherwise
                error(['unrecognized prior ' prior])
            end
            obj.w = w;
        end


        function pred = predict(obj,varargin)
        % Predict the class labels of the specified test examples using the
        % specified method.
        %
        % FORMAT:
        %          [pred,samples] = predict(model, 'name1', val1, 'name2', val2, ...)
        %
        % INPUT:
        %
        % 'X'      The test data: X(i,:) is the ith case
        %
        %
        % 'method' {['plugin'] | 'mc' | 'integral'}
        % 
        %           plugin   - predictions are made using the MAP estimates,
        %                      (default)
        %           mc       - only available if w is an MvnDist object, which
        %                      will be true if this model was fit with 
        %                      method = 'bayesian'.
        %           integral - only available in 2-class problems where w is an
        %                      MvnDist object. 
        %
        % nsamples [1000] The number of Monte Carlo samples to perform. Only
        %                 used when method = 'mc'
        %
        % OUTPUT:
        %
        % pred    - is a series of discrete distributions over class labels,
        %           one for each test example X(i,:). All of these are
        %           represented in a single DiscreteDist object such that
        %           pred.probs(i,c) is the probability that example i
        %           belongs to class c. 
        %           
        %           If method = 'mc', pred is a SampleDistDiscrete object storing one
        %           distribution, (represented by samples) for every test
        %           example such that pred.samples(s,c,i) is the
        %           probability that example i is in class c according to sample
        %           s. Simply take the mode to obtain predicted class labels. 
            
            if(nargin == 2 && ~ischar(varargin{1}))
                varargin = [varargin,varargin{1}];
                varargin{1} = 'X';
            end
            
            [X,method,nsamples] = process_options(varargin,'X',[],'method','plugin','nsamples',1000);
            if ~isempty(obj.transformer)
                X = test(obj.transformer, X);
            end
            w = obj.w; 
            if(isempty(w)),error('Call fit() first or specify w');end
            switch method
                
                case 'plugin'
                    if(isa(w,'ConstDist')),w = w.point; end
                    if(isa(w,'MvnDist')),w = w.mu;end
                    pred = DiscreteProductDist(multiSigmoid(X,w(:)),obj.classSupport);  %#ok
                case 'mc'
                    if(~isa(w,'MvnDist')),
                        error('w must be an MvnDist object to draw Monte Carlo samples. Either specify p(w|D) as an mvnDist or call fit with ''prior'' = ''l2'', ''method'' = ''bayesian''');
                    end
                    Wsamples = sample(w,nsamples);
                    samples = zeros(nsamples,obj.nclasses,size(X,1));
                    for s=1:nsamples
                        samples(s,:,:) = multiSigmoid(X,Wsamples(s,:)')';
                    end
                    pred = SampleDistDiscrete(samples,obj.classSupport);
                case 'integral'
                    if(obj.nclasses ~=2),error('This method is only available in the 2 class case');end
                    if(~isa(w,'MvnDist')),
                        error('w must be an MvnDist object for this method. Either specify p(w|D) as an mvnDist or call fit with ''prior'' = ''l2'', ''method'' = ''bayesian''');
                    end
                    p = sigmoidTimesGauss(X, w.mu(:), w.Sigma);
                    p = p(:);
                    pred = DiscreteProductDist([p,1-p],obj.classSupport);
                otherwise
                    error('%s is an unsupported prediction method',method); 
            end        
        end

        function p = logprob(obj, X, y)
        % p(i) = log p(y(i) | X(i,:), obj.w), y(i) in 1...C
            pred = predict(obj,'X',X,'method','plugin');
            P = pred.mu;
            Y = oneOfK(y, obj.nclasses);
            p =  sum(sum(Y.*log(P)));
        end

    end


    methods(Access = 'protected')

      function w = fitL1(obj,X,Y1,lambda,optMethod,offsetAdded)
        % Fit using the specified L1 regularizer, lambda via the specified
        % method.
        [n,d] = size(X);                                %#ok
        lambdaVec = lambda*ones(d,obj.nclasses-1);
        if(offsetAdded),lambdaVec(:,1) = 0;end
        lambdaVec = lambdaVec(:);
        options.verbose = false;
        w = zeros(d*(obj.nclasses-1),1);
        [w,fEvals] = L1General(optMethod, @multinomLogregNLLGradHessL2,gradFunc,w,...
          lambda,Vec, options, X, Y1, 0, false);
      end



    end

%%   
    methods(Static = true)

        function testClass()
        % check functions are syntactically correct
            n = 10; d = 3; C = 2;
            X = randn(n,d );
            y = sampleDiscrete((1/C)*ones(1,C), n, 1);
            mL2 = LogregDist('nclasses', C);
            mL2 = fit(mL2, 'X', X, 'y', y,'method','bayesian');
            predMAPL2 = predict(mL2, 'X',X);                                                %#ok
            [predMCL2]  = predict(mL2,'X',X,'method','mc','nsamples',2000);       %#ok
            predExactL2 = predict(mL2,'X',X,'method','integral');                           %#ok
            llL2 = logprob(mL2, X, y);                                                      %#ok
            %%
            mL1 = LogregDist('nclasses',C);
            mL1 = fit(mL1,'X',X,'y',y,'prior','L1','lambda',0.1);
            pred = predict(mL1,'X',X);                                                      %#ok
        end 

    end



end