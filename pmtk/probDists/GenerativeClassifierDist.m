classdef GenerativeClassifierDist < ProbDist

    
    properties
        classConditionals;      % a cell array of distributions 1 per class
        classPrior;             % Discrete or Discrete_Dirichlet 
        transformer;            % a data transformer object, (e.g. pcaTransformer)
    end
    
    methods
        
        function obj = GenerativeClassifierDist(varargin)
          % GenerativeClassifierDist(transformer, classConditionals,
          % classPrior)
            [obj.transformer,obj.classConditionals,obj.classPrior] = processArgs(varargin,...
                '-transformer'           ,[],...
                '-classConditionals'     ,[],...
                '-classPrior'            ,[]);
        end
              
        
        function obj = fit(obj,varargin)
            % Fit the classifier
            %
            % FORMAT:
            %           obj = fit(obj,'name1',val1,'name2',val2,...)
            % INPUT:
            %           'X'            - X(i,:) is i'th case
            %           'y'            - the training labels
            [X,y] = process_options(varargin,'X',[],'y',[]);
            obj.classPrior = fit(obj.classPrior, '-data',  colvec(y));
            if(~isempty(obj.transformer))
                [X,obj.transformer] = train(obj.transformer,X);
            end
            for c=1:length(obj.classConditionals)
              Xc = X(y==obj.classPrior.support(c),:);
              obj.classConditionals{c} = fit(obj.classConditionals{c},'-data',Xc);
            end
        end
        
        function pred = predict(obj,X)
            % pred(i) = p(y|X(i,:), params), a DiscreteDist
            if(~isempty(obj.transformer))
                X = test(obj.transformer,X);
            end
            %logpy = log(predict(obj.classPrior));
            logpy = log(pmf(obj.classPrior));
            n= size(X,1);
            C = length(obj.classConditionals);
            L = zeros(n,C);
            for c=1:C
                L(:,c) = sum(logprob(obj.classConditionals{c},X),2) + logpy(c);
            end;
            post = exp(normalizeLogspace(L));
            pred = DiscreteDist('-T', post', '-support', obj.classPrior.support);
        end
        
        function d = ndimensions(obj)
            d = ndimensions(obj.classConditionals{1});
        end
        
    end
  
    
  
end

