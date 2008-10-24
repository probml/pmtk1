classdef crossValidation
    % Perform k-fold cross validation on a specified test function given a range of
    % CVvalues, (e.g. lambdas) and a specified loss function. Supports
    % multi-dimensional CV, i.e. CV over two or more variables.
    %
    % Requires Kfold.m, oneStdErrorRule.m, pdfcrop, gridSpace
    % Version 1.0 (October 22, 2008)

    properties(GetAccess = 'public',SetAccess = 'private')  % Input Properties
        %%
        testFunction;
        % A function handle of the form
        % @(Xtrain,ytrain,Xtest,cvValue)f(Xtrain,ytrain,Xtest,cvValue).
        %
        % The values returned by testFunction are passed directly to the loss
        % function and must be appropriate, i.e. predicted output if using MSE,
        % etc.
        %
        % A test function is usually created by forming a function handle, which
        % combines fit and predict functions as in the following:
        % testFunction = @(Xtrain,ytrain,Xtest,lambda)...
        %    predict(fit(Xtrain,ytrain,lambda,fitOptions),Xtest,predictOptions);
        %
        % If you want to cross validate over 2 or more values simultaneously,
        % i.e. you want to find the best combination of say lambda and sigma,
        % testFunction must take in an additional input as in
        % testfunction =...
        % @(Xtrain,ytrain,Xtest,lambda,sigma)f(Xtrain,ytrain,Xtest,lambda,sigma)
        % See CVvalues for more information.
        %%
        lossFunction;
        % [Default: 'MSE']
        % Either a string, (e.g. 'MSE', 'ZeroOne') or a function handle that
        % takes as input,(1) the first output of testFunction and, (2) the true
        % target labels from each fold as in lossFunction =
        % @(yhat,ytest)sum(abs(yhat - ytest)) where yhat is the output of
        % testFunction(Xtrain,Ytrain,Xtest,lambda). The output must be scalar.
        %%
        CVvalues;
        % The values to cross validate on, (i.e. the lambdas).
        % This is either a row vector of values, e.g. logspace(-3,0,20) or a
        % cell array of row vectors in which case CV is performed over a grid,
        % testing the Cartesian product of the two or more sets. In such a case,
        % bestValue will be a vector, e.g. [lambda,sigma], where [lambda,sigma]
        % is the best combination from the two sets w.r.t. the specified loss
        % function.
        %%
        Xdata;       % An n-by-d matrix of all of the X data available, (crossValidation will automatically partition into folds)
        Ydata;       % An n-by-k matrix of all of the Y data available, (crossValidation will automatically partition into folds)
        nfolds;      % [DEFAULT: 5]    The number of folds to use
        %%
        % [DEFAULT: true] If true, the CV value(s) appearing first on the list
        % of examined values, that is within one standard error of the is
        % chosen. In multidimensional CV, the order is determined via ndgrid.
        use1SDrule;
        %%
        verbose;     % [Default: true] If true, the results of each evaluation are written to the console
        doplot;      % [Default: true] (Ignored if cross validating over more than 2 dimensions).
        %              If true, a CV curve is plotted.
    end

    properties(GetAccess = 'public',SetAccess = 'private')  % Ouptut Properties
        CVvaluesReshaped;        
        %%
        % All of the CV values, reshaped in the order in which they are
        % evaluated. This only differs from CVvalues when multidimensional CV is
        % performed. The entries in the following properties correspond to rows
        % of CVvaluesReshaped.
        %%
        lossFunctionOutput;
        % A matrix of size nvalues-by-nfolds where nvalues is the number of
        % CVvalues considered. Each entry is the loss function value evaluated
        % using the corresponding cross validation value(s), (from
        % CVvaluesReshaped) and data from the corresponding fold.  When 2d CV is
        % performed for example, nvalues = numel(lambdaVals)*numel(sigmaVals).
        %%
        meanLoss;      % Simply mean(lossFunctionOutput,2), i.e. mean accross all of the folds
        SEloss;        % The standard error of lossFunctionOutput accross all of the folds
        bestNDX;       % row index into CVvaluesReshaped corresponding to the CV chosen values
        %%
        bestValue;
        % The best value(s) determined by CV. In the case of multidimensional
        % CV, bestValue is a vector, e.g. [bestLambda,bestSigma], etc. 
        %%
        bestLossValue;  % The loss value associated with the chosen CVvalue(s).
    end

   

    methods
        
        function obj = crossValidation(varargin)
        % Class constructor
            [   obj.testFunction         ,...
                obj.lossFunction         ,...
                obj.CVvalues             ,...
                obj.Xdata                ,...
                obj.Ydata                ,...
                obj.nfolds               ,...
                obj.use1SDrule           ,...
                obj.verbose              ,...
                obj.doplot           ] =  ...
                process_options(varargin ,...
                'testFunction' ,[]       ,...
                'lossFunction' ,'MSE'    ,...
                'CVvalues'     ,[]       ,...
                'Xdata'        ,[]       ,...
                'Ydata'        ,[]       ,...
                'nfolds'       ,5        ,...
                'use1SDrule'   ,false    ,...
                'verbose'      ,true     ,...
                'doplot'       ,true     );
            
            if(obj.verbose),tic,end
            if(iscell(obj.CVvalues))
                obj.CVvaluesReshaped = num2cell(gridSpace(obj.CVvalues{:}));
            else
                if(size(obj.CVvalues,1) < size(obj.CVvalues,2))
                    obj.CVvalues = obj.CVvalues';
                end
                obj.CVvaluesReshaped = num2cell(obj.CVvalues);
            end

            if(ischar(obj.lossFunction))
                switch obj.lossFunction
                    case 'MSE'
                        obj.lossFunction = @(yhat,ytest)mse(reshape(yhat,size(ytest)),ytest);
                    case 'ZeroOne'
                        obj.lossFunction = @(yhat,ytest)sum(reshape(yhat,size(ytest)) ~= ytest);
                    otherwise
                        error('Invalid Loss Function');
                end
            end
            obj.errorCheck();
            obj = run(obj);
        end

        function h = plot(obj)
            % Plot the CV curve
            ndms = size(obj.CVvaluesReshaped,2);
            if(ndms > 2)
                fprintf('Cannot plot in more than 2 dimensions\n');
                return;
            end
            if(ndms == 1)
                h = plot2d(obj);
            elseif(ndms == 2)
                h = plot3d(obj);
            end
        end

    end

    methods(Access = 'protected')

        function obj = run(obj)
        % Actually run the cross validation
            if(obj.verbose)
                t = toc;
                str = sprintf('Fold: %d of %d\nElapsed Time: %d minutes, %d seconds',1,obj.nfolds,floor(t/60),floor(rem(t,60)));
                wbar = waitbar(0,str);
            end
            [n,d] = size(obj.Xdata);                                    %#ok
            [trainfolds,testfolds] = Kfold(n,obj.nfolds,false);
            nvalues = size(obj.CVvaluesReshaped,1);
            obj.lossFunctionOutput = zeros(nvalues,n);
            for f = 1:obj.nfolds
                Xtrain = obj.Xdata(trainfolds{f},:);
                ytrain = obj.Ydata(trainfolds{f},:);
                Xtest =  obj.Xdata(testfolds{f},:);
                ytest =  obj.Ydata(testfolds{f},:);
                for v = 1:nvalues
                    if(obj.verbose)
                       t = toc;
                       str = sprintf('Fold: %d of %d\nElapsed Time: %d minutes, %d seconds',f,obj.nfolds,floor(t/60),floor(rem(t,60)));
                       waitbar( (v+(f-1)*nvalues)/(obj.nfolds*nvalues),wbar,str); 
                    end
                    val = obj.CVvaluesReshaped(v,:);
                    testOutput = obj.testFunction(Xtrain,ytrain,Xtest,val{:});
                    lossValue = obj.lossFunction(testOutput,ytest);
                    obj.lossFunctionOutput(v,testfolds{f}) = lossValue;
                    if(obj.verbose)
                        vstr = mat2str(cell2mat(val),5);
                        vstr = [vstr,blanks(max(10-length(vstr),1))]; %#ok
                        fprintf('value(s): %s\t loss: %f\n',vstr,lossValue);
                    end
                end
            end
            obj.meanLoss = mean(obj.lossFunctionOutput,2);
            obj.SEloss = std(obj.lossFunctionOutput,[],2)/sqrt(n);

            if(obj.use1SDrule)
                obj.bestNDX =  oneStdErrorRule(obj.meanLoss, obj.SEloss);
                obj.bestLossValue = obj.meanLoss(obj.bestNDX);
            else
                [obj.bestLossValue,obj.bestNDX] = min(obj.meanLoss);
            end
            bestValue = obj.CVvaluesReshaped(obj.bestNDX,:);
            obj.bestValue = cell2mat(bestValue);
            if(obj.verbose)
                close(wbar);
                fprintf('\n\nBest Value: %s, Loss: %f\n',mat2str(obj.bestValue,5),obj.bestLossValue);
                t = toc;
                fprintf('\nElapsed Time: %d minutes, %d seconds\n\n',floor(t/60),floor(rem(t,60)));
            end
            obj.CVvaluesReshaped = cell2mat(obj.CVvaluesReshaped);
            if(obj.doplot)
                if(size(obj.CVvaluesReshaped,2) < 3)
                    plot(obj);
                end
            end

        end
        
        
        function h = plot2d(obj)
           h = figure;
           errorbar(obj.CVvalues,obj.meanLoss,obj.SEloss,'LineWidth',2);
           title('Cross Validation Curve');
           xlabel('CVvalues');
           ylabel('Loss Function Value');
           ax = axis;
           line([obj.bestValue,obj.bestValue],[ax(3),ax(4)],'LineStyle','--','Color','r');
           box on;
           axis tight;
           pdfcrop 
        end
        
        function h = plot3d(obj)
           h = figure; hold on;
           nrows = numel(obj.CVvalues{1});
           ncols = numel(obj.CVvalues{2});
           X = reshape(obj.CVvaluesReshaped(:,1),nrows,ncols);
           Y = reshape(obj.CVvaluesReshaped(:,2),nrows,ncols);
           Z = reshape(obj.meanLoss,nrows,ncols); 
           surf(X,Y,Z);
           view([0,90])
           colorbar;
           xlabel('first CV value');
           ylabel('second CV value');
           title('Color Denotes Loss Function Value');
           plot(obj.bestValue(1),obj.bestValue(2),'.k','LineWidth',2,'MarkerSize',10);
           axis tight;
        end

    end

    methods(Static = true)

        
        function demo()
            %% Cross Validation Over a 2D Grid of Values
            % Here we demonstrate how to cross validate two values, lambda and sigma
            % simultaneously using the crossValidation class. We use the crabs data set and
            % perform l2 logistic regression with an RBF expansion.
            %%
            % This model selection class requires that two key functions be specified: a loss
            % function, such as zero-one loss, mean squared error, nll, etc, and test function
            % of the following form:
            %%
            %  @(Xtrain,ytrain,Xtest,lambda,sigma)testFunction(Xtrain,ytrain,Xtest,lambda,sigma)
            %%
            % If we were cross validating over a single value, we would omit the "sigma". We
            % can cross validate over an n-dimensional grid just as easily, e.g.
            %%
            %  @(Xtrain,ytrain,Xtest,lambda,sigma)testFunction(Xtrain,ytrain,Xtest,lambda,sigma,eta,gamma)
            %%
            % However, the problem soon becomes computationally intractable.
            %
            % The test function will be evaluated at each fold for each combination of
            % values and its output will be passed directly to the loss function. In our
            % example, we will use zero-one loss and thus our test function must the return
            % predicted labels.
            %% Create the Test Function
            % Creating the test function will usually amount to composing fit and predict
            % functions together. This is what we do here, however because of the number of
            % options available and the RBF preprocessing, this will be a more advanced
            % example.
            %
            %%
            % One approach is to write a stand alone function with the right behavior. Here
            % is an example.
            %%
            %  function yhat = testFunction(Xtrain,ytrain,Xtest,lambda,sigma)
            %      T = chainTransformer({standardizeTransformer(false),kernelTransformer(sigma)});
            %      m = multinomLogregDist('nclasses',2,'transformer',T);
            %      m = fit(m,'X',Xtrain,'y',ytrain,'lambda',lambda,'prior','l2');
            %      pred = predict(m,Xtest);
            %      yhat = mode(pred);
            %  end
            %
            %%
            % We would then create the function handle as follows:
            % tfunc = @testFunction;
            %
            %% Use Anonymous Functions
            % However, it is often convenient to create the test function on the fly using
            % anonymous functions.
            %
            % The test function is the composition m( p( f( c( t( s , k ) ) ) ) where
            %%
            % * m is mode()
            % * p is fit()
            % * c is the model constructor
            % * t is the chain transformer constructor
            % * s is the standardizeTransformer constructor
            % * k is the kernalTransformer constructor
            %%
            % Our five input variables are defined in these functions as follows:
            %%
            % * f(Xtrain,ytrain,lambda)
            % * p(Xtest)
            % * k(sigma)
            %%
            % To make the composition clearer we will curry our functions, however,
            % this is not strictly necessary.
            %
            %%
            m = @mode;
            p = @(model,Xtest)predict(model,Xtest);
            f = @(model,Xtrain,ytrain,lambda)fit(model,'X',Xtrain,'y',ytrain,'lambda',lambda,'prior','l2');
            c = @(trans)multinomLogregDist('nclasses',2,'transformer',trans);
            t = @(a,b)chainTransformer({a(),b()});  % Use () to force evaluation before passing on
            s = @(x)standardizeTransformer(false);
            k = @(sigma)kernelTransformer('rbf',sigma);
            %%
            % Now let us compose c,t,s,k
            c = @(sigma)c(t(s(),k(sigma)));
            %%
            % Finally we compose the remaining functions.
            testFunction = @(Xtrain,ytrain,Xtest,lambda,sigma)m(p(f(c(sigma),Xtrain,ytrain,lambda),Xtest));
            %%
            % Of course we could have done this all in one step.
            %%
            %  testFunction = @(Xtrain,ytrain,Xtest,lambda,sigma)...
            %  mode(predict(fit(multinomLogregDist...
            %  'nclasses',2,'transformer',...
            %  chainTransformer(...
            %  {standardizeTransformer(false),kernelTransformer('rbf', sigma)})),...
            %  'X',Xtrain,'y',ytrain,'lambda',lambda,'prior','l2'),Xtest));
            %
            %% Create the Loss Function
            % We are now ready to perform the cross validation. We pass in a cell
            % array with test values for lambda and sigma and specify a loss function.
            % MSE and Zero-One are built in and can be used by specifying a string
            % instead of a function handle, such as 'MSE', or 'ZeroOne'. Here we use
            % zero one loss but use a function handle for demonstration purposes.
            %
            %%
            lossFunction = @(yhat,ytest)sum(reshape(yhat,size(ytest))~=ytest);
            %% Perform the Cross Validation
            load crabs;
            %%
            % Performing the actual cross validation simply amounts to instantiating the
            % class with the right inputs.
            modelSelection = crossValidation(                     ...
                'testFunction' , testFunction                    ,...
                'CVvalues'     , { logspace(-5,0,50) , 1:0.5:15 },... % every combination will be tested
                'lossFunction' , lossFunction                    ,...
                'verbose'      , false                           ,... % true by default - shows progress
                'Xdata'        , Xtrain                          ,...
                'Ydata'        , ytrain                          );

            bestVals = modelSelection.bestValue;
            bestLambda = bestVals(1)                                  %#ok
            bestSigma  = bestVals(2)                                  %#ok
            set(gca,'XScale','log');
            %% Plot Results
            % By default, a figure of the cross validation curve is plotted, (at least in 2D
            % and 3D). Here we transform the x-axis since our lambda values were log-spaced.
            %%
            %  set(gca,'XScale','log');
            %% Refit
            % Now lets retrain the model using the best lambda and sigma values
            T = chainTransformer({standardizeTransformer(false),kernelTransformer('rbf',bestSigma)});
            m = multinomLogregDist('nclasses',2,'transformer',T);
            m = fit(m,'X',Xtrain,'y',ytrain,'lambda',bestLambda,'prior','l2');
            pred = predict(m,Xtest);
            yhat = mode(pred);
            errorRate = mean(yhat'~=ytest)
            %%
            % Notice that we correctly classify all of the examples.


        end



    end



    methods(Access = 'private')

        function errorCheck(obj)
        % Perform basic error checking.     
            if(isempty(obj.testFunction))
                error('You must include a handle to a test function of the form @(Xtrain,Ytrain,Xtest,lambda,...)')
            end
            if(isempty(obj.CVvalues))
                error('You must specify a range of CVvalues to assess via cross validation');
            end
            if(isempty(obj.Xdata) || isempty(obj.Ydata))
                error('You must specify X and Y data');
            end
        end

    end
end

