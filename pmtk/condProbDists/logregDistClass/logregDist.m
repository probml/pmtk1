classdef logregDist < condProbDist 
% logistic regression, multiclass conditional distribution

    properties
        w;                      % w is the posterior distribution of the weights. 
                                % The form depends on how this object was fit. 
                                % If method = 'map', the default, then w 
                                % represents the MAP estimate and is stored as a
                                % constDist object. If method = 'bayesian', then
                                % w is an mvnDist representing the laplace
                                % approximation to the posterior. 
                                
        transformer;            % A data transformer object, e.g. kernelTransformer
        
        nclasses;               % The number of classes
    end

    %% Main methods
    methods

        function m =logregDist(varargin)
        % Constructor
            [m.transformer,  m.w, m.nclasses] = process_options( varargin ,...
                'transformer', []             ,...
                'w'          , []             , ...
                'nclasses'   , []);
            
            if(~isempty(m.w) && isnumeric(m.w))
                m.w = constDist(m.w);
            end
        end

        function [obj, output] = fit(obj, varargin)
        % Compute the posterior distribution over w, the weights. This is either
        % a delta distribution representing the MAP estimate if method =
        % 'map', (the default), or a full mvnDist distribution representing
        % the laplace approximation to the posterior, if method = 'bayesian'.
        %
        % FORMAT:
        %           model = fit(model, 'name1', val1, 'name2', val2, ...)
        % INPUT:
        %
        % 'X'      - The training examples: X(i,:) is the ith case
        % 'y'      - The class labels for X in {1...C}
        % 'prior'  - {'L1' | 'L2' | ['none']}]
        % 'lambda' - [0] regularization value
        % 'method  - {['map'] | 'bayesian'}   The latter is unsupported in the
        %                                     case that prior = 'l1'
        % 'optMethod' - 
        %
        %               ----L1----
        % {['projection'] | 'iteratedridge' | 'grafting' | 'orthantwise' |  'pdlb' |   'sequentialqp'
        %  |'subgradient' | 'unconstrainedapx' |  'unconstrainedapxsub' |  'boundoptrelaxed' |
        %  'boundoptstepwise'}
        %
        %               ---- L2----
        % {['lbfgs'] | 'newton' | 'bfgs' | 'newton0' | 'netwon01bfgs' | 'cg' |
        % 'bb' | 'sd' | 'tensor' | 'boundoptrelaxed' | 'boundoptstepwise'}
        %
        % OUTPUT:
        %
        % obj      - The fitted logregDist object
        % output   - A structure holding the output of the fitting algorithm, if any.
            
            [X, y,  prior, lambda, method,optMethod] = process_options(varargin,...
                'X'            , []                 ,...
                'y'            , []                 ,...
                'prior'        , 'none'             ,...
                'lambda'       , 0                  ,...
                'method'       , 'map'           ,...
                'optMethod'    , 'default'           );

            output = [];
            if lambda > 0 && strcmpi(prior, 'none'), prior = 'L2'; end

            offsetAdded = false;
            if ~isempty(obj.transformer)
                [X, obj.transformer] = train(obj.transformer, X);
                offsetAdded = obj.transformer.addOffset();
            end

            if isempty(obj.nclasses), obj.nclasses = length(unique(y)); end
            Y1 = oneOfK(y, obj.nclasses);

            switch lower(prior)
                case {'l1'}
                     obj = fitL1(obj,X,Y1,lambda,method,optMethod,offsetAdded);
                case {'l2', 'none'}
                    [obj,output] = fitL2(obj,X,Y1,lambda,method,optMethod,offsetAdded);
                otherwise
                    error(['unrecognized prior ' prior])
            end
        end

        function [pred,samples] = predict(obj,varargin)
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
        % 'w'      (1) unspecified, then then obj.w is used instead. 
        %        OR
        %          (2) a matrix of weights of size ndimensions-by-(nclasses-1)
        %        OR
        %          (3) a constDist object centered on the map estimate of the
        %              posterior p(w|D)
        %        OR
        %          (4) an mvnDist object representing the posterior p(w|D)
        %
        % 'method' {['plugin'] | 'mc' | 'integral'}
        % 
        %           plugin   - predictions are made using the MAP estimates,
        %                      (default)
        %           mc       - only available if w is an mvnDist object, which
        %                      will be true if this model was fit with 
        %                      method = 'bayesian'.
        %           integral - only available in 2-class problems where w is an
        %                      mvnDist object. 
        %
        % nsamples [1000] The number of Monte Carlo samples to perform. Only
        %                 used when method = 'mc'
        %
        % OUTPUT:
        %
        % pred    - is a series of discrete distributions over class labels,
        %           one for each test example X(i,:). All of these are
        %           represented in a single discreteDist object such that
        %           pred.probs(i,c) is the probability that example i
        %           belongs to class c. If method = 'mc', pred is the result of
        %           averaging over all of the samples. 
        %
        % samples - empty, [], unless method = 'mc'
        %           a sampleDist object storing one distribution, (represented
        %           by samples) for every test example such that
        %           sampleDist.samples(s,c,i) is the probability that example i
        %           is in class c according to sample s. In particular, pred
            
            if(nargin == 2 && ~ischar(varargin{1}))
                varargin = [varargin,varargin{1}];
                varargin{1} = 'X';
            end
            
            [X,w,method,nsamples] = process_options(varargin,'X',[],'w',[],'method','plugin','nsamples',1000);
            if ~isempty(obj.transformer)
                X = test(obj.transformer, X);
            end
            samples = [];
            if(isempty(w)), w = obj.w; end
            if(isempty(w)),error('Call fit() first or specify w');end
            switch method
                
                case 'plugin'
                    if(isa(w,'constDist')),w = w.point; end
                    if(isa(w,'mvnDist')),w = w.mu;end
                    pred = discreteDist(multiSigmoid(X,w(:)));          %#ok
                case 'mc'
                    if(~isa(w,'mvnDist')),
                        error('w must be an mvnDist object to draw Monte Carlo samples. Either specify p(w|D) as an mvnDist or call fit with ''prior'' = ''l2'', ''method'' = ''bayesian''');
                    end
                    Wsamples = sample(w,nsamples);
                    samples = zeros(nsamples,obj.nclasses,size(X,1));
                    for s=1:nsamples
                        samples(s,:,:) = multiSigmoid(X,Wsamples(s,:)')';
                    end
                    samples = sampleDist(samples);
                    pred = discreteDist(mean(samples));
                case 'integral'
                    if(obj.nclasses ~=2),error('This method is only available in the 2 class case');end
                    if(~isa(w,'mvnDist')),
                        error('w must be an mvnDist object for this method. Either specify p(w|D) as an mvnDist or call fit with ''prior'' = ''l2'', ''method'' = ''bayesian''');
                    end
                    p = sigmoidTimesGauss(X, w.mu(:), w.Sigma);
                    p = p(:);
                    pred = discreteDist([p,1-p]);
                otherwise
                    error('%s is an unsupported prediction method',method);
                  
            end        
        end

        function p = logprob(obj, X, y)
        % p(i) = log p(y(i) | X(i,:), obj.w), y(i) in 1...C
            pred = predict(obj,'X',X,'method','plugin');
            P = pred.probs;
            Y = oneOfK(y, obj.nclasses);
            p =  sum(sum(Y.*log(P)));
        end

    end


    methods(Access = 'protected')

        function obj = fitL1(obj,X,Y1,lambda,method,optMethod,offsetAdded)
        % Fit using the specified L1 regularizer, lambda via the specified method.
            
            if(~strcmpi(method,'map'))
                error('%s method is not currently supported given an L1 prior',method);
            end
            [n,d] = size(X);                                %#ok
            
            lambdaVec = lambda*ones(d,obj.nclasses-1);
            if(offsetAdded),lambdaVec(:,1) = 0;end
            lambdaVec = lambdaVec(:);
            options.verbose = false;
            
            optfunc = [];
            switch lower(optMethod)
                case 'iteratedridge'
                    optfunc = @L1GeneralIteratedRige;
                case 'projection'
                    options.order = -1;
                    optfunc = @L1GeneralProjection;
                case 'grafting'
                    optfunc = @L1GeneralGrafting;
                case 'orthantwise'
                    optfunc = @L1GeneralOrthantWist;
                case 'pdlb'
                    optfunc = @L1GeneralPrimalDualLogBarrier;
                case 'sequentialqp'
                    optfunc = @L1GeneralSequentialQuadraticProgramming;
                case 'subgradient'
                    optfunc = @L1GeneralSubGradient;
                case 'unconstrainedapx'
                    optfunc = @L1GeneralUnconstrainedApx;
                case 'unconstrainedapxsub'
                   optfunc = @L1GeneralUnconstrainedApx_sub;
                case 'boundoptrelaxed'
                    if(offsetAdded),warning('logregDist:offset','currently penalizes offset weight'),end
                    [w,output] =  compileAndRun('boundOptL1overrelaxed',X, Y1, lambda);
                    output.ftrace = output.ftrace(output.ftrace ~= -1);
                case 'boundoptstepwise'
                    if(offsetAdded),warning('logregDist:offset','currently penalizes offset weight'),end
                    [w, output] = compileAndRun('boundOptL1Stepwise',X, Y1, lambda);
                    output.ftrace = output.ftrace(output.ftrace ~= -1);    
                otherwise
                    options.order = -1; 
                    optfunc = @L1GeneralProjection;
            end
            if(~isempty(optfunc))
                w = optfunc(@multinomLogregNLLGradHessL2,zeros(d*(obj.nclasses-1),1),lambdaVec,options,X,Y1,0,false);
            end
            obj.w = constDist(w);
        end

        function [obj,output] = fitL2(obj,X,Y1,lambda,method,optMethod,offsetAdded)
        % Fit using the specified L1 regularizer, lambda via the specified method.
           [n,d] = size(X);                                                         %#ok
            switch lower(optMethod)
                case 'boundoptrelaxed'
                    if(offsetAdded),warning('logregDist:offset','currently penalizes offset weight'),end
                    [w, output] = compileAndRun('boundOptL2overrelaxed',X, Y1, lambda);
                    output.ftrace = output.ftrace(output.ftrace ~= -1);
                case 'boundoptstepwise'
                    if(offsetAdded),warning('logregDist:offset','currently penalizes offset weight'),end
                    [w, output] = compileAndRun('boundOptL2Stepwise',X, Y1, lambda);
                    output.ftrace = output.ftrace(output.ftrace ~= -1);
                case 'fminuncnewton'
                     winit = zeros(d,1);
                     options = optimset('Display','none','Diagnostics','off','GradObj','on','Hessian','on');
                     [w,output] = fminunc(@multinomLogregNLLGradHessL2, winit, options, X, Y1, lambda);
                otherwise
                    if(strcmpi(optMethod,'default'))
                        optMethod = 'lbfgs';
                    end
                    options.Method = optMethod;
                    options.Display = false;
                    winit = zeros(d*(obj.nclasses-1),1);
                    [w, f, exitflag, output] = minFunc(@multinomLogregNLLGradHessL2, winit, options, X, Y1, lambda,offsetAdded); %#ok
            end
            
            switch method
                
                case 'map'
                    obj.w = constDist(w);
                case 'bayesian'
                    try
                        [nll, g, H] = multinomLogregNLLGradHessL2(w, X, Y1, lambda,offsetAdded); %#ok  H = hessian of neg log lik    
                        C = inv(H);
                        obj.w = mvnDist(w, C); %C  = inv Hessian(neg log lik)
                    catch
                        warning('logregDist:Laplace','Laplace approximation to the posterior could not be computed because the Hessian could not be inverted...using MAP estimate instead');
                        obj.w = constDist(w);
                    end
                otherwise
                    error('%s method is not currently supported given an L2 prior',method);
            end
        end

    end

%%   
    methods(Static = true)

        function testClass()
        % check functions are syntactically correct
            n = 10; d = 3; C = 2;
            X = randn(n,d );
            y = sampleDiscrete((1/C)*ones(1,C), n, 1);
            mL2 = logregDist('nclasses', C);
            mL2 = fit(mL2, 'X', X, 'y', y,'method','bayesian');
            predMAPL2 = predict(mL2, 'X',X);                                                %#ok
            [predMCL2,samplesL2]  = predict(mL2,'X',X,'method','mc','nsamples',2000);       %#ok
            predExactL2 = predict(mL2,'X',X,'method','integral');                           %#ok
            llL2 = logprob(mL2, X, y);                                                      %#ok
            %%
            mL1 = logregDist('nclasses',C);
            mL1 = fit(mL1,'X',X,'y',y,'prior','L1','lambda',0.1);
            pred = predict(mL1,'X',X);                                                      %#ok
        end 

    end



end