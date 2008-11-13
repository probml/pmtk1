classdef ModelSelection
% This is a general purpose model selection class. It can be used to perform any
% score based model selection such as cross validation, bic, aic, etc. This
% generality is achieved by modularizing the model selection process and
% allowing the user to specify custom functions, which are executed at the
% the appropriate points. Default built in functions exist for many of these. 
%
% See modelSelectionDemo under the PMTK/examples directory for a number of
% examples and read constructor documentation below. 
%
% Here is a schematic of the model selection process. The arrows represent
% parameter passing. 
%
%                  ____________________________________________________________
%                 |                                                            |
%                 |                         ____________________________       |
%                 |                        |                            |      |
%   __________    |  __________       _____V____      _________      ___|___   |
%  |  select  |<--- |  search  |<--- |  score   |--->|  test   |--->| loss  |  |
%  |__________|   | |__________|---> |__________|    |_________|    |_______|  |
%                 |                                                            |
%                 |                search until stopping criteria reached      |
%                 |____________________________________________________________|
% 
%
% The search function proposes a model and asks the score function to score it,
% this score is returned to search and the process repeated until search decides
% to stop. Search then passes the results to select, which may choose say the
% model with the best score or the simplest within one standard error of best,
% etc. 
%
% Score has at its disposal two additional functions, which it can optionally
% use. Suppose that the scoring function were cross validation, then a possible
% loss function would be mean squared error and the test function would take in
% Xtrain,ytrain,Xtest, and the model parameters, and return predictions, which
% would be passed to loss. In the case of cv, the scoring function would perform
% nfold loops before returning the score for a single model. 





    properties(GetAccess = 'public',SetAccess = 'private')  % Input Properties
        searchFunction;
        scoreFunction;
        testFunction;
        lossFunction;
        selectFunction;
        models;
        ordering;
        Xdata;      
        Ydata;       
    end

    properties(GetAccess = 'public', SetAccess = 'private')
        CVnfolds;      
        verbose;     
        doplot;
        progressBar;
    end
    
    properties(GetAccess = 'public',SetAccess = 'private')  
        sortedResults;
        bestModel;
    end

    methods
        
        function obj = ModelSelection(varargin)
        % Constructing an object of this class with the right inputs 
        % automatically performs model selection. 
        %
        % FORMAT: 
        %
        %  obj = ModelSelection('name1',val1,'name2',val2,...);
        %
        % INPUT:
        %  
        % searchFunction  - By default, the built in exhaustive search function
        %                   is used and this does not have to be specified. 
        %
        %                   If you want to use your own custom search function,
        %                   this must be a function handle, which takes in an
        %                   object of type ModelSelection as input and uses the
        %                   values stored in its properties. In particular, it
        %                   must minimally call obj.scoreFunction for every
        %                   model it will search over, passing the model and the
        %                   ModelSelection object. Score will then return the
        %                   model's score and these results must be stored and
        %                   returned in an array of structs with the fields
        %                   'model','score', and 'stdErr'. See the implemented
        %                   exhaustiveSearch method for an example. You should
        %                   also update the progress bar each iteration. 
        %
        % scoreFunction   - By default, the built in cvScore function, (for
        %                   cross validation) is used and this does not have to 
        %                   be specified. The CVnfolds property stores the
        %                   number of folds to perform. 
        %
        %                   If you want to use your own custom score function,
        %                   this must be a function handle to a function that
        %                   takes as input, the ModelSelection object as well as
        %                   a single model to be scored as passed by the search
        %                   function. The function must then return a scalar 
        %                   score for the model and the standard error of this
        %                   score, (use 0 if this is not a meaningful quantity).
        %
        % testFunction     - This is handle to a function, which is
        %                    optionally used by scoreFunction and must have
        %                    an appropriate interface. Its output will usually
        %                    be passed to lossFunction. 
        %
        %                    In the case of cross validation, (default), it must 
        %                    have the following interface:
        %
        %                    @(Xtrain,ytrain,Xtest,param1,param2,...) 
        %
        %                    where param1,param2,... etc collectively represent
        %                    the model, e.g. perhaps lambda and sigma. The
        %                    scoring function passes these parameters by
        %                    emptying the ith model into the testFunction using
        %                    the cell operation {:} as in 
        %
        %                    testFunction(Xtrain,ytrain,Xtest,models{i}{:})
        %
        %                    If, for instance, the ith model contains three
        %                    values, all three of these are passed to
        %                    testFunction as three separate parameters. 
        %
        % lossFunction     - This is a function handle - optionally used
        %                    by the scoreFunction, which will generally pass in
        %                    the ouput of testFunction. In the case of cross
        %                    validation, this is set automatically but may need
        %                    to be overridden. By default, mean squared error is
        %                    used if Ydata ~= round(Ydata), otherwise, zero-one
        %                    loss is used. If you want to use the default
        %                    cvScore function but your own lossFunction, note
        %                    that cvScore will pass it three inputs in this
        %                    order: (1) the output from testFunction, (2) Xtest,
        %                    (3) ytest. Your function must take these three
        %                    inputs even if it doesn't use them all. For
        %                    instance the built in mse loss function has this
        %                    form:
        %                    @(yhat,Xtest,ytest)mse(reshape(yhat,size(ytest)),ytest)
        %                    
        %
        % selectFunction   - This is a handle to a function, which takes in the
        %                    results of the model search and makes the final
        %                    decision as to the best model. The default function
        %                    simply chooses the model with the lowest score. Its
        %                    interface is as follows:
        %                    [bestModel,sortedResults] =  selector(obj,results)
        %                    where obj is the ModelSelection object and results
        %                    is an array of structs with the fields,
        %                    'model','score', and 'stdErr'. It returns the same
        %                    array of structs but sorted by score and the chosen
        %                    model. 
        % 
        % models           - These are the candidate models. Exhaustive search
        %                    uses these but custom search methods may or may
        %                    not. A model is defined as an arbitrary sized cell 
        %                    array, e.g. {lambda, sigma, delta} and the
        %                    models are stored collectively as a cell array of
        %                    cell arrays, i.e. models{i} returns the ith
        %                    model as a cell array. You can use the static
        %                    method ModelSelection.formatModels() to help format
        %                    models. See its interface for details. 
        %
        % 'Xdata'          - All of the input data, where Xdata(i,:) is the ith
        %                    case. In the case of cross validation, this is
        %                    automatically partitioned into training and test
        %                    sets during each fold. 
        %
        % 'Ydata'          - All of the output data, where Ydata(i,:) is the ith
        %                    target. 
        %
        % 'ordering'          - ['ascend'] | 'descend' Used by the default selector
        %                    function. If 'ascend', the best model is the one
        %                    with the smallest score, if 'descend' its the other
        %                    way round. 
        %
        % 'CVnfolds'       - Only used by cross validation, this value specifies
        %                    the number of folds to perform, (default = 5).
        %
        % 'verbose'        - True | False, built in and custom functions
        %                    will/should use this to determine whether interim 
        %                    details are displayed as the model selection runs. 
        %
        % 'doPlot'         - True | False, if true a plot of the results is
        %                    displayed. 
        %                            
        % OUTPUT:
        %
        % obj      - the model selection object, after it has run to completion.
        %
        %            obj.bestModel stores the selected model 
        %           
        %            obj.sortedResults stores all of the results in an array of 
        %            structs with the fields, 'model','score', and 'stdErr'.
        %
            [   obj.searchFunction       ,...
                obj.scoreFunction        ,...
                obj.testFunction         ,...
                obj.lossFunction         ,...
                obj.selectFunction       ,...
                obj.models               ,...
                obj.Xdata                ,...
                obj.Ydata                ,...
                obj.ordering                ,...
                obj.CVnfolds             ,...
                obj.verbose              ,...
                obj.doplot           ] =  ...
                process_options(varargin ,...
                'searchFunction',@exhaustiveSearch               ,...
                'scoreFunction' ,@cvScore                        ,...
                'testFunction'  ,@(varargin)varargin{:}          ,...
                'lossFunction'  ,[]                              ,...
                'selectFunction',@bestSelector                   ,...
                'models'        ,[]                              ,...
                'Xdata'         ,[]                              ,...
                'Ydata'         ,[]                              ,...
                'ordering'      ,'ascend'                        ,...
                'CVnfolds'      ,5                               ,...
                'verbose'       ,true                            ,...
                'doplot'        ,true                            );
            
            if(isempty(obj.models)),error('You must specify models to do model selection');end
                
            if(isempty(obj.lossFunction) && ~isempty(obj.Ydata))
                if(isequal(obj.Ydata,round(obj.Ydata)))
                    obj.lossFunction = @(yhat,Xtest,ytest)sum(reshape(yhat,size(ytest)) ~= ytest);
                else
                    obj.lossFunction = @(yhat,Xtest,ytest)mse(reshape(yhat,size(ytest)),ytest);
                end
            end
            if(obj.verbose)
                obj.progressBar = waitbar(0,'Model Selection Progress');
                tic
            end
            results = obj.searchFunction(obj);
            [obj.bestModel,obj.sortedResults] = obj.selectFunction(obj,results);
            if(obj.verbose)
                close(obj.progressBar);
            end
            if(obj.doplot)
                plot(obj,results);
            end
            
            if(obj.verbose)
               fprintf('Best Model = ');
               display(obj.bestModel); 
            end
            
        end
        
    end

    methods(Access = 'protected')
    %% Built in default functions 
        
        function [bestModel,sortedResults] = bestSelector(obj,results)
        % Default built in model selector - it simply selects the model with the 
        % lowest score. 
        %
        % INPUT: 
        %
        % obj      - the modelSelection object (read only)
        % results  - the results struct with fields: 'model','score', and
        %            'stdErr'
            [val,perm] = sort([results.score],obj.ordering);
            sortedResults = results(perm);
            bestModel = sortedResults(1).model; 
        end
        
        function results = exhaustiveSearch(obj)
        % Default built in search function, which evaluates every model in models
        % according to the specified scoring function, scoreFcn stored in obj.
        %
        % INPUT:
        %
        % obj        - the modelSelection object, (read only)
        %
        % OUTPUT:
        %
        % results    - an array of structs with 'model',score',and 'stdErr' fields
        %
            results = struct('model',{},'score',{},'stdErr',{});
            nmodels = size(obj.models,1);
            for i=1:nmodels;
                if(obj.verbose)
                    t = toc;
                    str = sprintf('Model: %d of %d\nElapsed Time: %d minute(s), %d seconds',i,nmodels,floor(t/60),floor(rem(t,60))); 
                    waitbar(i/nmodels,obj.progressBar,str); 
                end
                m = obj.models{i};
                results(i).model = m;
                try
                    [results(i).score,results(i).stdErr] = obj.scoreFunction(obj,m);
                catch
                    results(i).score = obj.scoreFunction(obj,m);
                    results(i).stdErr = 0;
                end
                
            end
        end
        
        function [score,stdErr] = cvScore(obj,model)
        % Default built in scoring function, which returns the cross validation
        % score of a single model evaluated using the test and loss functions
        % stored in obj. 
        %
        % INPUT: 
        %
        % obj       -   the modelSelection object, (read only)
        %
        % model     -   a single model stored in a cell array that will be emptied
        %               into the test function, e.g. model{:}. This will usually be
        %               a sequence of values, e.g. {lambda,sigma}.
        % OUTPUT:
        %
        % score      -  the mean cv score over all of the runs
        %
        % stdErr     -  the standard deviation of the of the cv score over all of
        %               the runs.     
        %
            n = size(obj.Xdata,1);                                    
            [trainfolds,testfolds] = Kfold(n,obj.CVnfolds,true);
            scoreArray = zeros(n,1);
            for f = 1:obj.CVnfolds
                Xtrain = obj.Xdata(trainfolds{f},:); 
                ytrain = obj.Ydata(trainfolds{f},:);
                Xtest  = obj.Xdata(testfolds{f} ,:);
                ytest  = obj.Ydata(testfolds{f},:);
                scoreArray(testfolds{f}) = obj.lossFunction(obj.testFunction(Xtrain,ytrain,Xtest,model{:}),Xtest,ytest);
            end
            score = mean(scoreArray);
            stdErr = std (scoreArray)/sqrt(n);
        end
        
    end
    
    methods(Access = 'protected')
        
         
        function plot(obj,results)
        % Plot the model selection results.    
            is2d = true;
            is3d = true;
            for i=1:size(obj.models,1)
               n = numel(obj.models{i});
               if(n ~= 1),is2d = false;end
               if(n ~= 2),is3d = false;end
            end
            if(is2d && isnumeric(obj.models{1}{1}))
                plotErrorBars2d(obj,results);
            end
            
            if(is3d && isnumeric(obj.models{1}{1}))
               plot3d(obj,results); 
            end
        end 
        
        
        
        
        function plotErrorBars2d(obj,results)
        % An error bar plot of the model selection curve. 
           h = figure;
           models = cell2mat(vertcat(results.model));
           scores = vertcat(results.score);
           stdErrs = vertcat(results.stdErr);
           errorbar(models,scores,stdErrs,'LineWidth',2);
           title('Model Selection Curve');
           xlabel('Model');
           ylabel('Score');
           ax = axis;
           line([obj.bestModel{1},obj.bestModel{1}],[ax(3),ax(4)],'LineStyle','--','Color','r');
           box on;
           axis tight;
           if(isequal(logspace(log10(min(models)),log10(max(models)),numel(models))',models))
               warning('off','MATLAB:Axes:NegativeDataInLogAxis');
               set(gca,'Xscale','log'); 
           end
           pdfcrop  
        end
        
        function plot3d(obj,results)
        % A plot of the error/score surface    
           figure; hold on;
           ms = cell2mat(vertcat(results.model));
           nrows = numel(unique(ms(:,1)));
           ncols = numel(unique(ms(:,2)));
           X = reshape(ms(:,1),nrows,ncols);
           Y = reshape(ms(:,2),nrows,ncols);
           Z = reshape(vertcat(results.score),nrows,ncols); 
           surf(X,Y,Z);
           view([0,90])
           colorbar;
           xlabel('first model param');
           ylabel('second model param');
           title(sprintf('Color Denotes Score\nVal1: %f\nVal2: %f',obj.bestModel{1},obj.bestModel{2}));
           axis tight;    
           box on;
           warning('off','MATLAB:Axes:NegativeDataInLogAxis');
           if(isequal(logspace(log10(min(ms(:,1))),log10(max(ms(:,1))),numel(unique(ms(:,1))))',unique(ms(:,1))))
               set(gca,'Xscale','log'); 
           end
           if(isequal(logspace(log10(min(ms(:,2))),log10(max(ms(:,2))),numel(unique(ms(:,2))))',unique(ms(:,2))))
               set(gca,'Yscale','log'); 
           end
           
        end
        
    end
    
    methods(Static = true)
        
         function models = formatModels(varargin)
         % Helper function to prepare the model space for model selection. The
         % output of this function can be passed directly to the ModelSelection
         % constructor. 
         %
         % Suppose you want to cross validate over an n-dimensional grid of
         % values, simply pass in the test points for each dimension separately.
         %
         % Examples: 
         %
         % models = ModelSelection.formatModels(1:10,0.1:0.05:1,3:7,0:1)
         % models = ModelSelection.formatModels(logspace(-2,0,20),1:0.5:15);
             if(nargin == 1)
                space = varargin{1}';
             else
                space = gridSpace(varargin{:});
             end
             models = cell(size(space,1),1);
             for i=1:size(space,1)
                models{i} = num2cell(space(i,:));
             end
        end
        
        
        
        function testClass()
        % Simple test of this class    
            load prostate;
            
            baseModel = LinregDist('transformer',ChainTransformer({StandardizeTransformer(false),addOnesTransformer()}));
            models = ModelSelection.formatModels(logspace(-5,3,100));
       
            %% CV MSE loss    
            testFunction = @(Xtrain,ytrain,Xtest,lambda)mode(predict(fit(baseModel,'X',Xtrain,'y',ytrain,'lambda',lambda),Xtest));
            msCVmse = ModelSelection('testFunction',testFunction,'Xdata',Xtrain,'Ydata',ytrain,'models',models);
            
            yhat = mode(predict(fit(baseModel,'X',Xtrain,'y',ytrain,'lambda',msCVmse.bestModel{1}),Xtest));    
            cvFinalError = mse(yhat,ytest);
            title(sprintf('CV mse loss\nlambda = %f\nFinal MSE: %f',msCVmse.bestModel{1},cvFinalError));
            xlabel('lambda');
            %% CV NLL loss
            
            testFunction = @(Xtrain,ytrain,Xtest,lambda)fit(baseModel,'X',Xtrain,'y',ytrain,'lambda',lambda);
            lossFunction = @(fittedObj,Xtest,ytest)-logprob(fittedObj,Xtest,ytest);
            msCVnll = ModelSelection('testFunction',testFunction,'lossFunction',lossFunction,'Xdata',Xtrain,'Ydata',ytrain,'models',models);
            
            yhat = mode(predict(fit(baseModel,'X',Xtrain,'y',ytrain,'lambda',msCVnll.bestModel{1}),Xtest));    
            cvnllFinalError = mse(yhat,ytest);
            title(sprintf('CV NLL\nlambda = %f\nFinal MSE: %f',msCVnll.bestModel{1},cvnllFinalError));
            xlabel('lambda');
            
            %% BIC
            scoreFcn = @(obj,model) -bicScore(fit(LinregDist('transformer',ChainTransformer({StandardizeTransformer(false),addOnesTransformer()})),'X',obj.Xdata,'y',obj.Ydata,'lambda',model{1}),obj.Xdata,obj.Ydata,model{1});
            msBic = ModelSelection('Xdata',Xtrain,'Ydata',ytrain,'models',models,'scoreFunction',scoreFcn);
            
            yhat = mode(predict(fit(baseModel,'X',Xtrain,'y',ytrain,'lambda',msBic.bestModel{1}),Xtest));    
            bicFinalError = mse(yhat,ytest);
            title(sprintf('BIC\nlambda = %f\nFinal MSE: %f',msBic.bestModel{1},bicFinalError));
            xlabel('lambda');
            
            %% AIC
            scoreFcn = @(obj,model) -aicScore(fit(LinregDist('transformer',ChainTransformer({StandardizeTransformer(false),addOnesTransformer()})),'X',obj.Xdata,'y',obj.Ydata,'lambda',model{1}),obj.Xdata,obj.Ydata,model{1});
            msAic = ModelSelection('Xdata',Xtrain,'Ydata',ytrain,'models',models,'scoreFunction',scoreFcn);
            
            yhat = mode(predict(fit(baseModel,'X',Xtrain,'y',ytrain,'lambda',msAic.bestModel{1}),Xtest));    
            aicFinalError = mse(yhat,ytest);
            title(sprintf('AIC\nlambda = %f\nFinal MSE: %f',msAic.bestModel{1},aicFinalError));
            xlabel('lambda');
            
            placeFigures('square',false);
    

           %% 2D CV mse loss

           testFunction = @(Xtrain,ytrain,Xtest,lambda,sigma)...
               mode(predict(fit(LogregDist(...
               'nclasses',2,'transformer',...
               ChainTransformer(...
               {StandardizeTransformer(false),KernelTransformer('rbf', sigma)})),...
               'X',Xtrain,'y',ytrain,'lambda',lambda,'prior','l2'),Xtest));
        
           load crabs;
           models = ModelSelection.formatModels(logspace(-6,-3,20) , 1:0.5:15 );
           
           ms = ModelSelection(                     ...
               'testFunction' , testFunction                    ,...
               'models'       , models,...  
               'Xdata'        , Xtrain                          ,...
               'Ydata'        , ytrain                          );
           
           bestVals = ms.bestModel;
           bestLambda = bestVals{1}
           bestSigma  = bestVals{2}
           
           T = ChainTransformer({StandardizeTransformer(false),KernelTransformer('rbf',bestSigma)});
           m = LogregDist('nclasses',2,'transformer',T);
           m = fit(m,'X',Xtrain,'y',ytrain,'lambda',bestLambda,'prior','l2');
           pred = predict(m,Xtest);
           yhat = mode(pred);
           errorRate = mean(yhat'~=ytest)

           %% 2D CV nll loss
           
           
           %% 2D BIC
           
           scoreFcn = @(obj,model) -bicScore(fit(LinregDist('transformer',ChainTransformer({StandardizeTransformer(false),KernelTransformer('rbf',model{2})})),'X',obj.Xdata,'y',obj.Ydata,'lambda',model{1}),obj.Xdata,obj.Ydata,model{1});
           ms2dBIC = ModelSelection('scoreFunction',scoreFcn,'models',models,'Xdata',Xtrain,'Ydata',ytrain);
           bestVals = ms2dBIC.bestModel;
           bestLambda = bestVals{1}
           bestSigma  = bestVals{2}
           T = ChainTransformer({StandardizeTransformer(false),KernelTransformer('rbf',bestSigma)});
           m = LogregDist('nclasses',2,'transformer',T);
           m = fit(m,'X',Xtrain,'y',ytrain,'lambda',bestLambda);
           pred = predict(m,Xtest);
           yhat = mode(pred);
           errorRate = mean(yhat'~=ytest)
           
           
           
           
           
           
           
           
           
           
           
  
         
        
        
        end
        
        
        
        
        
    end
        
 end

