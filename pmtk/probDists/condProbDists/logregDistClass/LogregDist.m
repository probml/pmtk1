classdef LogregDist < CondProbDist 
%% Logistic Regression, Multiclass Conditional Distribution

    properties
        w;                      
        transformer;            % A data transformer object, e.g. KernelTransformer
        nclasses;               % The number of classes
        classSupport;           % The suppport of the target y, e.g. [0,1], [-1,+1], 1:K, etc. 
        priorStrength;
        prior;
        optMethod;
    end

    
    
    
    %% Main methods
    methods

        function m = LogregDist(varargin)
        % Constructor
            [m.transformer,  m.w, m.nclasses, m.prior, m.priorStrength, m.optMethod] = ...
              process_options( varargin ,...
                'transformer', []             ,...
                'w'          , []             , ...
                'nclasses'   , [], ...
                'prior', 'none', ...
                'priorStrength', [], ...
                'optMethod', 'default');
              
              %error('deprecated')
        end

        function [obj, output] = fit(obj, varargin)
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
                'prior'        , obj.prior          ,...
                'priorStrength', obj.priorStrength  ,...
                'optMethod'    , obj.optMethod           );
            output = [];
            if(isempty(lambda))
                lambda = 0;
            end
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
                objective = @(w,junk) multinomLogregNLLGradHessL2(w, X, Y1,0,false);
                options.verbose = false;
                if(strcmpi(optMethod,'default'))
                  optMethod = 'L1GeneralProjection';
                  options.order = -1; % significant speed improvement with this setting
                end
                [w,fEvals] = L1General(optMethod, objective, winit,lambdaVec, options);
              case {'l2', 'none'}
                objective = @(w,junk) multinomLogregNLLGradHessL2(w, X, Y1,lambda,offsetAdded);
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


        function pred = predict(obj,X)
          % pred(i) = p(y|X(i,:), w)
            if ~isempty(obj.transformer)
                X = test(obj.transformer, X);
            end
            pred = DiscreteDist('-T', multiSigmoid(X,obj.w(:))','-support',obj.classSupport);      
        end

        function p = logprob(obj, X, y)
          % p(i) = log p(y(i) | X(i,:), obj.w), y(i) in 1...C
          pred = predict(obj,X);
          P = pmf(pred)';
          Y = oneOfK(y, obj.nclasses);
          p =  sum(sum(Y.*log(P)));
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
    mL2 = fit(mL2, 'X', X, 'y', y,'prior','L2','priorStrength',0.1);
    predMAPL2 = predict(mL2,X);
    llL2 = logprob(mL2, X, y);
    mL1 = LogregDist('nclasses',C);
    mL1 = fit(mL1,'X',X,'y',y,'prior','L1','priorStrength',0.1);
    pred = predict(mL1,X);
  end

end



end