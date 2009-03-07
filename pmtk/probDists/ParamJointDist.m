classdef ParamJointDist < ParamDist
    
    % Parametric joint distribution, which supports inference
    % about states of some components (dimensions) given evidence on others.
    % We keep track of what has been conditioned on here.
    % (Abstract class)
    
    properties
        infEng;
        conditioned = false;
        % conditioned =  true if infEng has been initialized
        % by calling condition. We remember the inserted data.
        visVars;
        visVals;
        domain;
    end
    
    methods
        
        function model = condition(model, visVars, visValues)
          error('deprecated');
            % enter evidence that visVars=visValues
            if nargin < 2
                visVars = []; visValues = [];
            end
            model.conditioned = true;
            model.visVars = visVars;
            model.visVals = visValues;
            [model.infEng] = condition(model.infEng, model, visVars, visValues);
        end
        
        function [postQuery,model] = marginal(model, queryVars,varargin)
          error('deprecated')
            % postQuery = p(queryVars) conditional on the most recent
            % condition operation
            if ~model.conditioned
                model = condition(model);
            end
            if(nargin == 1)
                [postQuery,model.infEng] = marginal(model.infEng);
            else
                [postQuery,model.infEng] = marginal(model.infEng, queryVars,varargin{:});
            end
            
        end
        
        function [samples] = sample(model, n)
            if ~model.conditioned, model = condition(model); end
            if nargin < 2, n = 1; end
            [samples] = sample(model.infEng,  n);
        end
        
        function L = logprob(model, X)
          error('deprecated')
            % L(i) = log p(X(i,:) | params), where columns are hidden nodes
            if ~model.conditioned, model = condition(model); end
            %L = logprob(model.infEng, X, normalize);
            %XX = insertVisData(model,X);
            L = logprobUnnormalized(model, X) - lognormconst(model);
        end
        
        function XX = insertVisData(model, X)
            % X contains values of hidden variables
            % We insert the data that we have already conditioned on
            if(isstruct(model.domain))
                error('Not yet supported for models with multiple types of variables');
            end
            [n nhid] = size(X);
            nvis = length(model.visVars);
            if nvis == 0, XX = X; return; end
            XX = zeros(n, nhid+nvis);
            domain = model.domain;
            V = lookupIndices(model.visVars, domain);
            hidVars = setdiffPMTK(domain, model.visVars);
            H = lookupIndices(hidVars, domain);
            nv = size(model.visVals,1);
            if nv==1 && n>1
                XX(:, V) = repmat(model.visVals, n, 1);
            else
                XX(:, V) = model.visVals;
            end
            XX(:, H) = X;
        end
        
        %function logZ = logprobUnnormalized(model, X)
        %  %if ~model.conditioned, model = condition(model); end
        %  logZ = logprobUnnormalized(model.infEng, X);
        %end
        
        function logZ = lognormconst(model)
            %if ~model.conditioned, model = condition(model); end
            if(~ismember('lognormconst',methods(model.infEng)))
                error('The current inference engine does not support this operation');
            end
            logZ = lognormconst(model.infEng);
        end
        
        function mu = mean(model)
            if(~ismember('mean',methods(model.infEng)))
                error('The current inference engine does not support this operation');
            end
            if ~model.conditioned, model = condition(model); end
            mu = mean(model.infEng);
        end
        
        function mu = mode(model)
            if(~ismember('mode',methods(model.infEng)))
                error('The current inference engine does not support this operation');
            end
            if ~model.conditioned, model = condition(model); end
            mu = mode(model.infEng);
        end
        
        function C = cov(model)
            if(~ismember('cov',methods(model.infEng)))
                error('The current inference engine does not support this operation');
            end
            if ~model.conditioned, model = condition(model); end
            C = cov(model.infEng);
        end
        
        function C = var(model)
            if(~ismember('var',methods(model.infEng)))
                error('The current inference engine does not support this operation');
            end
            if ~model.conditioned, model = condition(model); end
            C = var(model.infEng);
        end
        
        function Xc = impute(model, X)
          error('deprecated')
            % Fill in NaN entries of X using posterior mode on each row
            [n] = size(X,1);
            Xc = X;
            for i=1:n
                hidNodes = find(isnan(X(i,:)));
                if isempty(hidNodes), continue, end;
                visNodes = find(~isnan(X(i,:)));
                visValues = X(i,visNodes);
                tmp = condition(model, visNodes, visValues);
                postH = marginal(tmp, hidNodes);
                %postH = predict(obj, visNodes, visValues);
                Xc(i,hidNodes) = rowvec(mode(postH));
            end
        end
        
        %{
        function [Xc,model] = emImpute(model,X)
            % Fill in NaN entries of X using the EM algorithm
            warning('buggy') % KPM
            [n] = size(X,1);
            Xc = X;
            
            oldlikelihood = -inf;
            newlikelihood = +inf;
            count = 1;
            while(~approxeq(oldlikelihood,newlikelihood,0.0001))
                
                % E Step
                for i=1:n
                    hidNodes = find(isnan(X(i,:)));
                    if isempty(hidNodes), continue, end;
                    visNodes = find(~isnan(X(i,:)));
                    visValues = X(i,visNodes);
                    tmp = condition(model,visNodes,visValues);
                    postH = marginal(tmp,hidNodes);
                    Xc(i,hidNodes) = rowvec(mode(postH));
                end
                % M step
                model = fit(model,'data',Xc);
                oldlikelihood = newlikelihood;
                newlikelihood = sum(logprob(model,Xc));
                count = count + 1;
                fprintf('iter: %d\n',count)
            end
        end
        %}
        
        
    end
    
end

