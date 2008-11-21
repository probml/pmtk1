classdef ModelSelection
% This is a general purpose model selection class. It can be used to perform any
% score based model selection such as cross validation, bic, aic, etc. This
% generality is achieved by modularizing the model selection process and
% allowing the user to specify custom functions, which are executed at the
% the appropriate points. Default built in functions exist for many of these. 
%
% See modelSelect1D and modelSelect2D under the PMTK/examples/modelSelect
% directory for a number of examples and read the constructor documentation below. 
%
% Here is a schematic of the model selection process. The arrows represent
% parameter passing. 
%
%                  ____________________________________________________________
%                 |                                                            |
%                 |                         ____________________________       |
%                 |                        |                            |      |
%   __________    |  __________       _____V____      _________      ___|___   |
%  |  select  |<--- |  search  |<--- |  score   |--->| predict |--->| loss  |  |
%  |__________|   | |__________|---> |__________|    |_________|    |_______|  |
%                 |                                                            |
%                 |                search until stopping criteria reached      |
%                 |____________________________________________________________|
% 
%
% The search function proposes a model and asks the score function to score it;
% this score is returned to search and the process repeated until search decides
% to stop. Search then passes the results to select, which may choose say the
% model with the best score or the simplest within one standard error of best,
% etc. 
%
% Score has at its disposal two additional functions, which it can optionally
% use: predict and loss. Suppose that the scoring function were cross
% validation, then a possible loss function would be mean squared error and the
% predict function would take in Xtrain,ytrain,Xtest, and the model parameters,
% and return predictions, which would be passed to loss. In the case of cv, the
% scoring function would perform nfold loops before returning the score for a
% single model. 


    properties(GetAccess = 'public',SetAccess = 'private')  % Input Properties
        searchFunction;
        scoreFunction;
        predictFunction;
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
        %                   this must be a handle to a function, which takes in an
        %                   object of type ModelSelection as input and uses the
        %                   values stored in its properties. In particular, it
        %                   must minimally call obj.scoreFunction for every
        %                   model it will search over, passing the model and the
        %                   ModelSelection object. Score will then return the
        %                   model's score and these results must be stored and
        %                   returned in an array of structs with the fields
        %                   'model','score', and 'stdErr'. See the implemented
        %                   exhaustiveSearch method for an example. You should
        %                   also update the progress bar each iteration. Note 
        %                   the 'model' field stores the index into obj.models. 
        %
        % scoreFunction   - By default, the built in cvScore function, (for
        %                   cross validation) is used and this does not have to 
        %                   be specified. The CVnfolds property stores the
        %                   number of folds to perform. 
        %
        %                   If you want to use your own custom score function,
        %                   this must be a handle to a function that takes as
        %                   input, the ModelSelection object as well as a single
        %                   model to be scored as passed by the search function.
        %                   The function must then return a scalar score for the
        %                   model and the standard error of this score, (use 0
        %                   if this is not a meaningful quantity).
        %
        % predictFunction  - This is a handle to a function, which is
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
        %                    emptying the ith model into the predictFunction using
        %                    the cell operation {:} as in 
        %
        %                    predictFunction(Xtrain,ytrain,Xtest,models{i}{:})
        %
        %                    If, for instance, the ith model contains three
        %                    values, all three of these are passed to
        %                    predictFunction as three separate parameters. 
        %
        % lossFunction     - This is a function handle - optionally used
        %                    by the scoreFunction. In the case of cross
        %                    validation, this is set automatically but may need
        %                    to be overridden. By default, mean squared error is
        %                    used if Ydata ~= round(Ydata), otherwise, zero-one
        %                    loss is used. If you want to use the default
        %                    cvScore function but your own lossFunction, note
        %                    that cvScore will pass it three inputs in this
        %                    order: (1) the output from predictFunction, (2) Xtest,
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
        %                    model. Note that the field model stores indices
        %                    into obj.models.
        % 
        % models           - These are the candidate models. Exhaustive search
        %                    uses these but custom search methods may or may
        %                    not. A model is defined as an arbitrary sized cell 
        %                    array, e.g. {lambda, sigma, delta} and the
        %                    models are stored collectively as a cell array of
        %                    cell arrays, i.e. models{i} returns the ith
        %                    model as a cell array. You can use the static
        %                    method ModelSelection.formatModels() to help create 
        %                    the model space. See its interface for details.
        %                    
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
        %                    with the smallest score, if 'descend' - the other
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
        %            obj.bestModel stores the best model. (In actual fact it
        %            stores the index to the best model but when queried, it 
        %            automatically returns the best model). 
        %           
        %            obj.sortedResults stores all of the results in an array of 
        %            structs with the fields, 'model','score', and 'stdErr'. The
        %            model field stores indices into obj.models. 
        %
        %% PROCESS INPUTS
            [   obj.searchFunction       ,...
                obj.scoreFunction        ,...
                obj.predictFunction         ,...
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
                'searchFunction' ,@exhaustiveSearch               ,...
                'scoreFunction'  ,@cvScore                        ,...
                'predictFunction',@(varargin)varargin{:}          ,...
                'lossFunction'   ,[]                              ,...
                'selectFunction' ,@bestSelector                   ,...
                'models'         ,[]                              ,...
                'Xdata'          ,[]                              ,...
                'Ydata'          ,[]                              ,...
                'ordering'       ,'ascend'                        ,...
                'CVnfolds'       ,5                               ,...
                'verbose'        ,true                            ,...
                'doplot'         ,true                            );
            
            %% ERROR CHECK
            if(isempty(obj.models)),error('You must specify models to do model selection');end
            
            %% SET DEFAULT LOSS
            if(isempty(obj.lossFunction) && ~isempty(obj.Ydata))
                if(iscell(obj.Ydata))
                    obj.lossFunction = @(yhat,Xtest,ytest)sum(canonizeLabels(yhat) ~= canonizeLabels(ytest));
                elseif(isequal(obj.Ydata,round(obj.Ydata)))
                    obj.lossFunction = @(yhat,Xtest,ytest)sum(reshape(yhat,size(ytest)) ~= ytest);
                else
                    obj.lossFunction = @(yhat,Xtest,ytest)mse(reshape(yhat,size(ytest)),ytest);
                end
            end
            %% SETUP PROGRESS BAR
            if(obj.verbose)
                obj.progressBar = waitbar(0,'Model Selection Progress');
                tic
            end
            %% RUN
            results = obj.searchFunction(obj);
            %% SELECT
            [obj.bestModel,obj.sortedResults] = obj.selectFunction(obj,results);
            %% DISPLAY
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
            %%
        end
        
        function best = get.bestModel(obj)
        % Intercept requests for the best model so that we can return the actual
        % best model when requested, not just its index.
            best = obj.models{obj.bestModel};
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
        %            'stdErr'. Note that 'model' stores indices into obj.models
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
        % results    - an array of structs with 'model',score',and 'stdErr'
        %              fields. The 'model' field stores indices into obj.models.
        %              
            results = struct('model',{},'score',{},'stdErr',{});
            nmodels = size(obj.models,1);
            for i=1:nmodels;
                if(obj.verbose)
                    t = toc;
                    str = sprintf('Model: %d of %d\nElapsed Time: %d minute(s), %d seconds',i,nmodels,floor(t/60),floor(rem(t,60))); 
                    waitbar((i-1)/nmodels,obj.progressBar,str); 
                end
                m = obj.models{i};
                results(i).model = i;
                try
                    [results(i).score,results(i).stdErr] = obj.scoreFunction(obj,m);
                catch
                    results(i).score = obj.scoreFunction(obj,m);
                    results(i).stdErr = 0;
                end
            end
            t = toc;
            str = sprintf('Finishing...\nElapsed Time: %d minute(s), %d seconds',floor(t/60),floor(rem(t,60))); 
            waitbar(1,obj.progressBar,str);
            pause(0.5);
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
                scoreArray(testfolds{f}) = obj.lossFunction(obj.predictFunction(Xtrain,ytrain,Xtest,model{:}),Xtest,ytest);
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
            try
            if(is2d && isnumeric(obj.models{1}{1}))
                plotErrorBars2d(obj,results);
            end
            
            if(is3d && isnumeric(obj.models{1}{1}))
               plot3d(obj,results); 
            end
            catch
                close;
            end
        end 
        
        function plotErrorBars2d(obj,results)
        % An error bar plot of the model selection curve. 
           h = figure;
           models = cell2mat(vertcat(obj.models{:}));%cell2mat(vertcat(results.model));
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
           fig = figure; hold on;
           ms = cell2mat(vertcat(obj.models{:}));%cell2mat(vertcat(results.model));
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
        
         function models = makeModelSpace(varargin)
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
    
    end
        
 end

