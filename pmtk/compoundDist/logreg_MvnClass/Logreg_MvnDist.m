classdef Logreg_MvnDist < CondProbDist 
%% Multinomial logistic Regression with Gaussian prior on weights

    properties
       wDist;
        transformer;            % A data transformer object, e.g. KernelTransfor
        nclasses;               % The number of classes
        classSupport;           % The suppport of the target y, e.g. [0,1], [-1,+1], 1:K, etc. 
        priorStrength;
        infMethod;
    end

    
    
    
    %% Main methods
    methods

        function m =Logreg_MvnDist(varargin)
        % Constructor
            [m.transformer,  m.wDist, m.priorStrength, m.infMethod, m.nclasses] = ...
              process_options( varargin ,...
                'transformer', []             ,...
                'wDist'          , []             , ...
                'priorStrength',[], ...
                'infMethod', 'laplace', ...
                'nclasses', []);
        end

        function [obj] = fit(obj, varargin)
         %
          % FORMAT:
          %           model = fit(model, 'name1', val1, 'name2', val2, ...)
          % INPUT:
          %
          % 'X'      - The training examples: X(i,:) is the ith case
          % 'y'      - The class labels for X in {1...C}
          % 'priorStrength'  - precision of Gaussian diagonal
          % 'infMethod' - {'laplace', 'mh'}
          
          [X, y, lambda, infMethod] = process_options(varargin,...
            'X'            , []                 ,...
            'y'            , []                 ,...
            'priorStrength'       , obj.priorStrength, ...
            'infMethod', obj.infMethod);
          offsetAdded = false;
          if ~isempty(obj.transformer)
            [X, obj.transformer] = train(obj.transformer, X);
            offsetAdded = obj.transformer.addOffset();
          end
          if isempty(obj.nclasses), obj.nclasses = length(unique(y)); end
          obj.ndimsX = size(X,2);
          obj.ndimsY = size(y,2);
        
          % First find mode
          tmp = LogregDist('prior', 'L2', 'priorStrength', lambda);
          tmp = fit(tmp, 'X', X, 'y', y);
          wMAP = tmp.w;
          % Then find Hessian at mode
          [Y1,obj.classSupport] = oneOfK(y, obj.nclasses);
          [nll, g, H] = multinomLogregNLLGradHessL2(wMAP, X, Y1, lambda,offsetAdded); %#ok  
          C = inv(H); %H = hessian of neg log lik    
          % Now find posterior
          switch infMethod
            case 'laplace', obj.wDist = MvnDist(wMAP, C);
            case 'mh', 
              d = length(wMAP);
              priorMu = zeros(d,1)';
              priorCov = (1/lambda)*eye(d);
              targetFn = @(w) logprob(LogregDist('w',w(:),'nclasses',obj.nclasses),X,y) + log(mvnpdf(w(:)',priorMu,priorCov));
              proposalFn = @(w) mvnrnd(w(:)',C);
              %initFn = @() mvnrnd(wMAP', 0.1*C);
              xinit = wMAP;
              samples = mhSample('symmetric', true, 'target', targetFn, 'xinit', xinit, ...
                'Nsamples', 1000, 'Nburnin', 100, 'proposal',  proposalFn);
              obj.wDist = SampleDist(samples);
            otherwise
              error(['unrecognized infMethod ' infMethod])
          end
          
        end

        function pred = predict(obj,X,varargin)
          % pred(i) = p(y|X(i,:)), a discreteDist
        %
        % 'X'      The test data: X(i,:) is the ith case
        %
        % 'method' - 
        %           'mc'       - monte carlo approximation
        %           integral - only available in 2-class problems
        %
        % nsamples [1000] The number of Monte Carlo samples to perform. Only
        %                 used when method = 'mc'
            [method,nsamples] = process_options(varargin,...
              'method','default','nsamples',1000);
            if ~isempty(obj.transformer)
                X = test(obj.transformer, X);
            end
            if strcmpi(method, 'default')
              if obj.nclasses==2 && isa(obj.wDist,'MvnDist')
                method = 'integral';
              else
                method = 'mc';
              end
            end
            switch method
              case 'mc'
                if isa(obj.wDist, 'MvnDist')
                  Wsamples = sample(obj.wDist,nsamples);
                else
                  Wsamples = obj.Wdist.samples;
                end
                n = size(X,1); C = obj.nclasses;
                P = zeros(n,C);
                for s=1:nsamples
                   P = P + multiSigmoid(X,Wsamples(s,:));
                end
                P = P / nsamples;
                pred = DiscreteDist('mu', P', 'support', obj.classSupport);
              case 'integral'
                if(obj.nclasses ~=2),error('This method is only available in the 2 class case');end
                if ~isa(obj.wDist,'MvnDist'), error('Only available for Gaussian posteriors'); end
                p = sigmoidTimesGauss(X, obj.wDist.mu(:), obj.wDist.Sigma);
                p = p(:);
                pred = BernoulliDist('mu',p);
              otherwise
                error('%s is an unsupported prediction method',method);
            end
        end

        function p = logprob(obj, X, y)
          % p(i) = log p(y(i) | X(i,:), obj.w), y(i) in 1...C
            pred = predict(obj,X);
            n = size(X,1);
            P = pred.mu;
            if size(pred.mu,2)==1
              P = [1-P, P];
            end
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
        mL2 = Logreg_MvnDist('nclasses', C, 'priorStrength', 1);
        mL2 = fit(mL2, 'X', X, 'y', y, 'infMethod', 'laplace');
        pred1 = predict(mL2, X, 'method', 'integral');
        pred2 = predict(mL2, X, 'method', 'mc');
        mL3 = fit(mL2, 'X', X, 'y', y, 'infMethod', 'mh');
        pred3 = predict(mL2, X);
        llL2 = logprob(mL2, X, y);
      end

    end



end