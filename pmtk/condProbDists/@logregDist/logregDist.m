classdef logregDist < condProbDist 
    % logistic regression, multiclass conditional distribution

    properties
        w; % MAP estimate of the optimal weights. w is  ndimensions-by-(nclasses-1)

        posteriorW;             % An mvnDist object representing the Laplace
                                % approximation to the full posterior, p(w|D).
                                % This is the approximation N(w|w_map|-inv(H))
                                % and is automatically set if fit() is called
                                % with prior = 'L2'.
        transformer;            % A data transformer object, e.g. kernelTransformer
        nclasses;               % The number of classes
    end

    %% Main methods
    methods

        function m =logregDist(varargin)
            % Constructor
            [transformer,  w, C] = process_options( varargin ,...
                'transformer', []             ,...
                'w'          , []             , ...
                'nclasses'   , []);

            m.transformer = transformer;
            m.w = w;
            m.nclasses = C;
        end

        function [obj, output] = fit(obj, varargin)
            % Compute the map estimate of the weights, obj.w for the model given
            % the training data, the prior type and regularization value, and the
            % optimization method. If prior is set to 'L2', the laplace
            % approximation to the posterior p(w|D) is also set: obj.posteriorW
            %
            % FORMAT:
            %           model = fit(model, 'name1', val1, 'name2', val2, ...)
            %
            % INPUT:
            %
            % 'X'      - The training examples: X(i,:) is the ith case
            % 'y'      - The class labels for X in {1...C}
            % 'prior'  - {'L1' | 'L2' | ['none']}]
            % 'lambda' - [0] regularization value
            % 'method' - 
            %
            %               ----L1----
            % {['projection'] | 'iteratedridge' | 'grafting' | 'orthantwise' |  'pdlb' |   'sequentialqp'
            %  |'subgradient' | 'unconstrainedapx' |  'unconstrainedapxsub' |  'boundoptrelaxed' |
            %  'boundoptstepwise'}
            %
            %            IF L2
            % {['lbfgs'] | 'newton' | 'bfgs' | 'newton0' | 'netwon01bfgs' | 'cg' |
            % 'bb' | 'sd' | 'tensor' | 'boundoptrelaxed' | 'boundoptstepwise'}
            %
            % OUTPUT:
            %
            % obj      - The fitted logregDist object
            % ouptut   - A structure holding the output of the fitting algorithm, if any.
            
            [X, y,  prior, lambda, method] = process_options(varargin,...
                'X'            , []                 ,...
                'y'            , []                 ,...
                'prior'        , 'none'             ,...
                'lambda'       , 0                  ,...
                'method'       ,'default'           );

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
                     obj = fitL1(obj,X,Y1,lambda,method,offsetAdded);
                case {'l2', 'none'}
                    [obj,output] = fitL2(obj,X,Y1,lambda,method,offsetAdded);
                otherwise
                    error(['unrecognized prior ' prior])
            end
        end

        function [pred,samples] = predict(obj,varargin)
        % Predict the class labels of the specified test examples using the
        % specified method.
        %
        % FORMAT:
        %          pred = predict(model, 'name1', val1, 'name2', val2, ...)
        %
        % INPUT:
        %
        % 'X'      The test data: X(i,:) is the ith case
        %
        % 'w'      (1) A matrix of weights of size
        %              ndimensions-by-(nclasses-1) representing the MAP
        %              estimate of the posterior p(w|D)
        %
        %          OR
        %
        %          (2) a mvnDist object representing the laplace
        %              approximation to the posterior p(w|D)
        %
        %          If not set, the value stored in this object is
        %          used instead, in particular: obj.w for case 1 and
        %          obj.posteriorW for case 2
        %
        % 'method' {['plugin'] | 'mc' | 'integral'}
        %
        %          The method to use:
        %
        %          'mc' is only available if w is an mvnDist object or if w was
        %          not specified and obj.posteriorW was set during the
        %          fitting of this object, (which occurs when the prior is
        %          set to 'L2').
        %
        %          'integral' uses a closed form approximation to the
        %          posterior predictive distribution but is only available
        %          in the 2 class case. Otherwise, the same conditions for
        %          'mc' apply
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
        %           belongs to class c.
        %
        % samples -  A 3D matrix such that samples(i,c,s) = probability that
        %            example i belongs to class c according to sample s. If
        %            method is not 'mc', s = 1 and samples = pred.probs
            [X,w,method,nsamples] = process_options(varargin,'X',[],'w',[],'method','plugin','nsamples',1000);
            if ~isempty(obj.transformer)
                X = test(obj.transformer, X);
            end
            
            switch method
                
                case 'plugin'
                    if(isempty(w)), w = obj.w; end
                    if(isempty(w)),error('Call fit() first or specify w');end
                    P = multiSigmoid(X,w);
                    pred = discreteDist(P);
                    samples = pred.probs;
                case 'mc'
                    w = checkW(w);
                    Wsamples = sample(w,nsamples);
                    samples = zeros(size(X,1),obj.nclasses,nsamples);
                    for s=1:nsamples
                        samples(:,:,s) = multiSigmoid(X,Wsamples(s,:)');
                    end
                    pred = discreteDist(mean(samples,3));
                case 'integral'
                    if(obj.nclasses ~=2),error('This method is only available in the 2 class case');                    end
                    w = checkW(w);
                    p = sigmoidTimesGauss(X, w.mu(:), w.Sigma);
                    pred = discreteDist([p,1-p]);
                otherwise
                    error('%s is an unsupported prediction method',method);
                  
            end
                    function w = checkW(w)
                        if(isempty(w)),w = obj.posteriorW;end
                        if(isempty(w)),error('Call fit() first with an L2 prior or specify an mvnDist object for w representing p(w|D)');end
                        if(~isa(w,'mvnDist')),error('w must be an mvnDist object for this method');end
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

        function obj = fitL1(obj,X,Y1,lambda,method,offsetAdded)
        % Fit using the specified L1 regularizer, lambda via the specified method.
            [n,d] = size(X);
            
            lambdaVec = lambda*ones(d,obj.nclasses-1);
            if(offsetAdded),lambdaVec(:,1) = 0;end
            lambdaVec = lambdaVec(:);
            options.verbose = false;
            optfunc = [];
            switch lower(method)
                case 'iteratedridge'
                    optfunc = @L1GeneralIteratedRige;
                case 'projection'
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
                    [obj.w,output] =  compileAndRun('boundOptL1overrelaxed',X, Y1, lambda);
                    output.ftrace = output.ftrace(output.ftrace ~= -1);
                case 'boundoptstepwise'
                    if(offsetAdded),warning('logregDist:offset','currently penalizes offset weight'),end
                    [obj.w, output] = compileAndRun('boundOptL1Stepwise',X, Y1, lambda);
                    output.ftrace = output.ftrace(output.ftrace ~= -1);    
                otherwise
                    optfunc = @L1GeneralProjection;
            end
            if(~isempty(optfunc))
                obj.w = optfunc(@multinomLogregNLLGradHessL2,zeros(d*(obj.nclasses-1),1),lambdaVec,options,X,Y1,0,false);
            end
        end

        function [obj,output] = fitL2(obj,X,Y1,lambda,method,offsetAdded)
        % Fit using the specified L1 regularizer, lambda via the specified method.
           [n,d] = size(X);
            switch lower(method)
                case 'boundoptrelaxed'
                    if(offsetAdded),warning('logregDist:offset','currently penalizes offset weight'),end
                    [obj.w, output] = compileAndRun('boundOptL2overrelaxed',X, Y1, lambda);
                    output.ftrace = output.ftrace(output.ftrace ~= -1);
                case 'boundoptstepwise'
                    if(offsetAdded),warning('logregDist:offset','currently penalizes offset weight'),end
                    [obj.w, output] = compileAndRun('boundOptL2Stepwise',X, Y1, lambda);
                    output.ftrace = output.ftrace(output.ftrace ~= -1);
                otherwise
                    if(strcmpi(method,'default'))
                        method = 'lbfgs';
                    end
                    options.Method = method;
                    options.Display = 0;
                    winit = zeros(d*(obj.nclasses-1),1);
                    [obj.w, f, exitflag, output] = minFunc(@multinomLogregNLLGradHessL2, winit, options, X, Y1, lambda,offsetAdded);
            end
            try
                wMAP = obj.w;
                [nll, g, H] = multinomLogregNLLGradHessL2(wMAP, X, Y1, lambda); % H = hessian of neg log lik
                C = inv(H);
                obj.posteriorW = mvnDist(wMAP, C); %C  = inv Hessian(neg log lik)
            catch
                warning('logregDist:Laplace','obj.posteriorW not set: could not invert the Hessian of the log-posterior.');
            end
        end

    end

%%

    %% Demos
    methods(Static = true)

        function test()
        % check functions are syntactically correct
            n = 10; d = 3; C = 2;
            X = randn(n,d );
            y = sampleDiscrete((1/C)*ones(1,C), n, 1);
            mL2 = logregDist('nclasses', C);
            mL2 = fit(mL2, 'X', X, 'y', y);
            predMAPL2 = predict(mL2, 'X',X);
            [predMCL2,samplesL2]  = predict(mL2,'X',X,'method','mc','nsamples',2000);
            predExactL2 = predict(mL2,'X',X,'method','integral');
            llL2 = logprob(mL1, X, y);
            %%
            mL1 = logregDist('nclasses',C);
            mL1 = fit(mL1,'X',X,'y',y,'prior','L1','lambda',0.1);
            pred = predict(mL1,'X',X);
        end

        function demoCrabs()
        % Here we fit an L2 regularized logistic regression model to the crabs 
        % data set and predict using three methods: MAP plugin approx, Monte
        % Carlo approx, and using a closed form approximation to the posterior
        % predictive. 
            [Xtrain, ytrain, Xtest, ytest] = makeCrabs;
            sigma2 = 32/5;
            T = chainTransformer({standardizeTransformer(false), kernelTransformer('rbf', sigma2)});
            m = logregDist('nclasses',2, 'transformer', T);
            lambda = 1e-3;
            m = fit(m, 'X', Xtrain, 'y', ytrain, 'lambda', lambda,'prior','l2');
            Pmap   = predict(m,'X',Xtest,'method','plugin');
            Pmc    = predict(m,'X',Xtest,'method','mc');
            Pexact = predict(m,'X',Xtest,'method','integral');
            nerrsMAP   = sum(mode(Pmap)' ~= ytest)
            nerrsMC    = sum(mode(Pmc)' ~= ytest)
            nerrsExact = sum(mode(Pexact)' ~= ytest)      
        end

        function demoOptimizer()
            setSeed(1);
            load soy; % n=307, d = 35, C = 3;
            %load car; % n=1728, d = 6, C = 3;
            methods = {'bb',  'cg', 'lbfgs', 'newton'};
            lambda = 1e-3;
            figure; hold on;
            [styles, colors, symbols] =  plotColors;
            for mi=1:length(methods)
                tic
                [m, output{mi}] = fit(logregDist, 'X', X, 'y', Y, ...
                    'lambda', lambda, 'method', methods{mi});
                T = toc
                time(mi) = T;
                w{mi} = m.w;
                niter = length(output{mi}.ftrace)
                h(mi) = plot(linspace(0, T, niter), output{mi}.ftrace, styles{mi});
                legendstr{mi}  = sprintf('%s', methods{mi});
            end
            legend(legendstr)
        end


        function demoVisualizePredictive()


            n = 300; d = 2;
            setSeed(0);
            X = rand(n,d);
            Y = ones(n,1);
            Y(X(:,1) < (0.4+0.1*randn(n,1))) = 2;
            Y(X(:,1) > (0.8 +0.05*randn(n,1))& X(:,2) > 0.8) = 2;
            Y(X(:,1) > 0.7 & X(:,1) < 0.8 & X(:,2) < 0.1) = 2;


            sigma2 = 1; lambda = 1e-3;
            T = chainTransformer({standardizeTransformer(false),kernelTransformer('rbf',sigma2)});
            model = logregDist('nclasses',2, 'transformer', T);
            model = fit(model,'prior','l2','lambda',lambda,'X',X,'y',Y,'method','lbfgs');


            [X1grid, X2grid] = meshgrid(0:0.01:1,0:0.01:1);
            [nrows,ncols] = size(X1grid);
            testData = [X1grid(:),X2grid(:)];
            pred = predict(model,testData);
            probGrid = reshape(pred.probs(:,1),nrows,ncols);


            figure;
            plot(X(Y==1,1),X(Y==1,2),'.r','MarkerSize',20); hold on;
            plot(X(Y==2,1),X(Y==2,2),'.b','MarkerSize',20);
            set(gca,'XTick',0:0.5:1,'YTick',0:0.5:1);
            title('Training Data');


            figure;
            surf(X1grid,X2grid,probGrid);
            shading interp;
            view([0 90]);
            colorbar
            set(gca,'XTick',0:0.5:1,'YTick',0:0.5:1);
            title('Predictive Distribution');



        end

        %{
function demoMnist()
      load('mnistALL')
      % train_images: [28x28x60000 uint8]
      % test_images: [28x28x10000 uint8]
      % train_labels: [60000x1 uint8]
      % test_labels: [10000x1 uint8]
      setSeed(0);
      Ntrain = 100;
      Ntest = 1000;
      Xtrain = zeros(10, Ntrain, 784);
      ytrain = zeros(10, Ntrain);
      Xtest = zeros(10, Ntrain, 784);
      ytest = zeros(10, Ntest);
      for c=1:10
        ndx = find(mnist.train_labels==c);
        ndx = ndx(1:Ntrain);
        Xtrain(c,:,:) = double(reshape(mnist.train_images(:,:,ndx), [28*28 length(ndx)]))';
        ytrain(c,:) = c*ones(Ntrain,1);
        ndx = find(mnist.test_labels==c);
        ndx = ndx(1:Ntest);
        Xtest(c,:,:) = double(reshape(mnist.test_images(:,:,ndx), [28*28 length(ndx)]))';
        ytest(c,:) = c*ones(Ntest,1);
      end
      Xtrain = reshape(Xtrain, 10*Ntrain, 784);
      ytrain = ytrain(:);
      Xtest = reshape(Xtest, 10*Ntest, 784);
      ytest = ytest(:);

      m = fit(logregDist, 'X', Xtrain, 'y', ytrain, 'lambda', 1e-3, 'prior', 'L2');
      pred = predict(m, Xtest);

    end
        %}
    end








    methods(Static = true)



        function demoSat()
            setSeed(1);
            stat = load('satData.txt'); % Johnson and Albert p77 table 3.1
            % stat=[pass(0/1), 1, 1, sat_score, grade in prereq]
            % where the grade in prereq is encoded as A=5,B=4,C=3,D=2,F=1
            y = stat(:,1);
            N = length(y);
            X = [stat(:,4)];
            T = addOnesTransformer;
            obj = logregDist('transformer', T);
            obj = fit(obj, 'X', X, 'y', y);

            % MLE
            figure; hold on
            [X,perm] = sort(X,'ascend');
            [py] = mean(predict(obj, X));
            y = y(perm);
            plot(X, y, 'ko', 'linewidth', 3, 'markersize', 12);
            plot(X, py, 'rx', 'linewidth', 3, 'markersize', 12);
            set(gca, 'ylim', [-0.1 1.1]);

            % Bayes
            obj = inferParams(obj, 'X', X, 'y', y, 'lambda', 1e-3);
            figure; hold on
            subplot(1,3,1); plot(obj.w); xlabel('w0'); ylabel('w1'); title('joint')
            subplot(1,3,2); plot(marginal(obj.w,1),'plotArgs', {'linewidth',2}); xlabel('w0')
            subplot(1,3,3); plot(marginal(obj.w,2),'plotArgs', {'linewidth',2}); xlabel('w1')

            figure; hold on
            n = length(y);
            S = 100;
            ps = postPredict(obj, X, 'method', 'MC', 'Nsamples', S);
            for i=1:n
                psi = marginal(ps, i);
                [Q5, Q95] = credibleInterval(psi);
                line([X(i) X(i)], [Q5 Q95], 'linewidth', 3);
                plot(X(i), median(psi), 'rx', 'linewidth', 3, 'markersize', 12);
            end
            plot(X, y, 'ko', 'linewidth', 3, 'markersize', 12);
            set(gca, 'ylim', [-0.1 1.1]);

            figure; hold on
            plot(X, y, 'ko', 'linewidth', 3, 'markersize', 12);
            for s=1:30
                psi = ps.samples(s,:);
                plot(X, psi, 'r-');
            end
        end

        function demoLaplaceGirolami()
            % Based on code written by Mark Girolami
            setSeed(0);
            % We generate data from two Gaussians:
            % x|C=1 ~ gauss([1,5], I)
            % x|C=0 ~ gauss([-5,1], 1.1I)
            N=30;
            D=2;
            mu1=[ones(N,1) 5*ones(N,1)];
            mu2=[-5*ones(N,1) 1*ones(N,1)];
            class1_std = 1;
            class2_std = 1.1;
            X = [class1_std*randn(N,2)+mu1;2*class2_std*randn(N,2)+mu2];
            y = [ones(N,1);zeros(N,1)];
            alpha=100; %Variance of prior (alpha=1/lambda)

            %Limits and grid size for contour plotting
            Range=8;
            Step=0.1;
            [w1,w2]=meshgrid(-Range:Step:Range,-Range:Step:Range);
            [n,n]=size(w1);
            W=[reshape(w1,n*n,1) reshape(w2,n*n,1)];

            Range=12;
            Step=0.1;
            [x1,x2]=meshgrid(-Range:Step:Range,-Range:Step:Range);
            [nx,nx]=size(x1);
            grid=[reshape(x1,nx*nx,1) reshape(x2,nx*nx,1)];

            % Plot data and plug-in predictive
            figure;
            m = fit(logregDist, 'X', X, 'y', y);
            plotPredictive(mean(predict(m,grid)));
            title('p(y=1|x, wMLE)')

            % Plot prior and posterior
            eta=W*X';
            Log_Prior = log(mvnpdf(W, zeros(1,D), eye(D).*alpha));
            Log_Like = eta*y - sum(log(1+exp(eta)),2);
            Log_Joint = Log_Like + Log_Prior;
            figure;
            J=2;K=2;
            subplot(J,K,1)
            contour(w1,w2,reshape(-Log_Prior,[n,n]),30);
            title('Log-Prior');
            subplot(J,K,2)
            contour(w1,w2,reshape(-Log_Like,[n,n]),30);
            title('Log-Likelihood');
            subplot(J,K,3)
            contour(w1,w2,reshape(-Log_Joint,[n,n]),30);
            title('Log-Unnormalised Posterior')
            hold

            %Identify the parameters w1 & w2 which maximise the posterior (joint)
            [i,j]=max(Log_Joint);
            plot(W(j,1),W(j,2),'.','MarkerSize',40);
            %Compute the Laplace Approximation
            tic
            m = inferParams(logregDist, 'X', X, 'y', y, 'lambda', 1/alpha, 'method', 'laplace');
            toc
            wMAP = m.w.mu;
            C = m.w.Sigma;
            %[wMAP, C] = logregFitIRLS(t, X, 1/alpha);
            Log_Laplace_Posterior = log(mvnpdf(W, wMAP', C)+eps);
            subplot(J,K,4);
            contour(w1,w2,reshape(-Log_Laplace_Posterior,[n,n]),30);
            hold
            plot(W(j,1),W(j,2),'.','MarkerSize',40);
            title('Laplace Approximation to Posterior')


            % Posterior predictive
            % wMAP
            figure;
            subplot(2,2,1)
            plotPredictive(mean(postPredict(m, grid, 'method', 'plugin')));
            title('p(y=1|x, wMAP)')

            subplot(2,2,2); hold on
            S = 100;
            plot(X(find(y==1),1),X(find(y==1),2),'r.');
            plot(X(find(y==0),1),X(find(y==0),2),'bo');
            pred = postPredict(m, grid, 'method', 'MC', 'nsamples', S);
            for s=1:min(S,20)
                p = pred.samples(s,:);
                contour(x1,x2,reshape(p,[nx,nx]),[0.5 0.5]);
            end
            set(gca, 'xlim', [-10 10]);
            set(gca, 'ylim', [-10 10]);
            title('decision boundary for sampled w')

            subplot(2,2,3)
            plotPredictive(mean(pred));
            title('MC approx of p(y=1|x)')

            subplot(2,2,4)
            plotPredictive(mean(postPredict(m, grid, 'method', 'integral')));
            title('numerical approx of p(y=1|x)')

            % subfunction
            function plotPredictive(pred)
                contour(x1,x2,reshape(pred,[nx,nx]),30);
                hold on
                plot(X(find(y==1),1),X(find(y==1),2),'r.');
                plot(X(find(y==0),1),X(find(y==0),2),'bo');
            end
        end

%         function demoOptimizer()
%             logregDist.helperOptimizer('documents');
%             logregDist.helperOptimizer('soy');
%         end

        function helperOptimizer(dataset)
            setSeed(1);
            switch dataset
                case 'documents'
                    load docdata; % n=900, d=600, C=2in training set
                    y = ytrain-1; % convert to 0,1
                    X = xtrain;
                    methods = {'bb',  'cg', 'lbfgs', 'newton'};
                case 'soy'
                    load soy; % n=307, d = 35, C = 3;
                    y = Y; % turn into a binary classification problem by combining classes 1,2
                    y(Y==1) = 0;
                    y(Y==2) = 0;
                    y(Y==3) = 1;
                    methods = {'bb',  'cg', 'lbfgs', 'newton',  'boundoptRelaxed'};
            end
            lambda = 1e-3;
            figure; hold on;
            [styles, colors, symbols] =  plotColors;
            for mi=1:length(methods)
                tic
                [m, output{mi}] = fit(logregDist, 'X', X, 'y', y, 'lambda', lambda, 'method', methods{mi});
                T = toc
                time(mi) = T;
                w{mi} = m.w;
                niter = length(output{mi}.ftrace)
                h(mi) = plot(linspace(0, T, niter), output{mi}.ftrace, styles{mi});
                legendstr{mi}  = sprintf('%s', methods{mi});
            end
            legend(legendstr)
        end





    end















end