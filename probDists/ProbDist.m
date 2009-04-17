classdef ProbDist
% This class represents an abstract proability distribution, e.g. a pdf or pmf.
% All PMTK probability distributions inherit directly or indirectly from ProbDist.
    
    
    %%  Main methods
    methods
      
        function nll = negloglik(obj,X)
            % The negative log likelihood of a data set
            % nll = -(1/n)sum_i(log p(X_i | params))
            % where X_i is the ith case, (e.g. X(i) or X(i,:) or X(:,:,i)) and n is
            % the number of cases.
            nll = -mean(logprob(obj, X),1);
        end
        
        function [mu, stdErr] = cvScore(obj, X, varargin)
            %CV Score using nll loss of the model.
            [nfolds,clamp] = process_options(varargin, 'nfolds', 5,'clamp',false);
            [n d] = size(X);
            [trainfolds, testfolds] = Kfold(n, nfolds);
            score = zeros(1,n);
            for k = 1:nfolds
                trainidx = trainfolds{k}; testidx = testfolds{k};
                Xtest = X(testidx,:);  Xtrain = X(trainidx, :);
                if(~clamp)
                    obj = fit(obj, 'data', Xtrain);
                end
                score(testidx) = negloglik(obj,  Xtest);
            end
            mu = mean(score);
            stdErr = std(score,0,2)/sqrt(n);
        end
        
        function bool = isTied(obj)
        % Should return true if any of the parameters are tied.    
           bool = false; 
        end
        
        function obj = clampTied(obj)
        % Clamp all tied parameters    
        end
        
        function obj = unclampTied(obj)
        % Unclamp all tied parameters    
        end
        
        function cellArray = copy(obj,varargin)
        % Create a cell array of copies of a probability distribution. Uses
        % repmat semantics - e.g. copy(obj,3) or copy(obj,3,5) or
        % copy(obj,[3,5,2])
        %
        % example:
        %
        % d = copy(DiscreteDist,3,5)  % - creates a 3-by-5 cell array of DiscreteDist objects
        %
           cellArray = num2cell(repmat(obj,varargin{:}));
        end
        
       
        
   
        
        
    end
    
    
    methods
        %% Plotting Methods
        function [h,p] = plot(obj, varargin)
        % plot a density function in 2d
        % handle = plot(pdf, 'name1', val1, 'name2', val2, ...)
        % Arguments are
        % xrange  - [xmin xmax] for 1d or [xmin xmax ymin ymax] for 2d
        % useLog - true to plot log density, default false
        % plotArgs - args to pass to the plotting routine, default {}
        % useContour - true to plot contour, false (default) to plot surface
        % npoints - number of points in each grid dimension (default 50)
        % eg. plot(p,  'useLog', true, 'plotArgs', {'ro-', 'linewidth',2})
            [xrange, useLog, plotArgs, useContour, npoints, scaleFactor] = process_options(...
                varargin, 'xrange', plotRange(obj), 'useLog', false, ...
                'plotArgs' ,{}, 'useContour', true, 'npoints', 100, 'scaleFactor', 1);
            if ~iscell(plotArgs), plotArgs = {plotArgs}; end
            if ndimensions(obj)==1
                xs = linspace(xrange(1), xrange(2), npoints);
                p = logprob(obj, xs(:));
                if ~useLog
                    p = exp(p);
                end
                p = p*scaleFactor;
                h = plot(colvec(xs), colvec(p), plotArgs{:});
            else
                [X1,X2] = meshgrid(linspace(xrange(1), xrange(2), npoints)',...
                    linspace(xrange(3), xrange(4), npoints)');
                [nr] = size(X1,1); nc = size(X2,1);
                X = [X1(:) X2(:)];
                p = logprob(obj, X);
                if ~useLog
                    p = exp(p);
                end
                p = reshape(p, nr, nc);
                if useContour
                    if~(any(isnan(p)))
                    [c,h] = contour(X1, X2, p, plotArgs{:});
                    end
                else
                    h = surf(X1, X2, p, plotArgs{:});
                end
            end
        end
        
        
    end
    
    methods
        function xrange = plotRange(obj, sf)
            if nargin < 2, sf = 3; end
            mu = mean(obj);
            if ndimensions(obj)==1
                s1 = sqrt(var(obj));
                x1min = mu(1)-sf*s1;   x1max = mu(1)+sf*s1;
                xrange = [x1min x1max];
            else
                C = cov(obj);
                s1 = sqrt(C(1,1));
                s2 = sqrt(C(2,2));
                x1min = mu(1)-sf*s1;   x1max = mu(1)+sf*s1;
                x2min = mu(2)-sf*s2; x2max = mu(2)+sf*s2;
                xrange = [x1min x1max x2min x2max];
            end
        end
    end
    
    
end