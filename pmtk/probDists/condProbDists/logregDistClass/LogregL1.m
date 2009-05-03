classdef LogregL1 < Logreg
%% Multinomial logistic regression with L1 prior on weights

    properties
      lambda;
    end  
    
    %% Main methods
    methods

     function m = LogregL1(varargin)
      % LogregL1(lambda, transformer, w, w0, nclasses, optMethod,
      % labelSpace)
      % optMethod can be any L1General method 
      % Default is projected gradient (first order, limited memory).
      % or 'boundoptOverrelaxed' or 'boundoptStepwise'
      % For details on the bound optimization methods, see
      % B. Krishnapuram et al, PAMI 2004
      % "Learning sparse Bayesian classifiers: multi-class formulation, 
      %     fast algorithms, and generalization bounds"
      [m.lambda, m.transformer, m.verbose, m.w, m.w0, m.nclasses,  m.optMethod,...
        m.labelSpace, m.addOffset] = ...
        processArgs( varargin ,...
        '-lambda', [], ...
        '-transformer', [], ...
         '-verbose', false, ...
        '-w'          , [], ...
        '-w0', [], ...
        '-nclasses'   , [], ...
        '-optMethod', 'projection', ...
        '-labelSpace', [], ...
        '-addOffset', true);
     end
    
    
       function df = dof(model)
         df = sum(abs(model.w(:)) ~= 0);  % num non zeros
       end
        
       
    end % methods
    
     methods(Access = 'protected')
       
      function [w, output, model] = fitCore(model, X, Y1,  winit)
        % Y1 is n*C (one of K)   
       switch model.optMethod
         % The boundopt code regularizes w0...
         case 'boundoptOverrelaxed'
           [w, output]  = boundOptL1overrelaxed(X, Y1, model.lambda);
         case 'boundoptStepwise',
            [w, output]  = boundOptL1stepwise(X, Y1, model.lambda);
         otherwise % minFunc
           d = size(X,2);
           C = model.nclasses;
           lambdaVec = model.lambda*ones(d,C-1);
           if model.addOffset,lambdaVec(:,1) = 0;end % don't regularize w0
           lambdaVec = lambdaVec(:);
           % unpenalized objective (lambda=0 turns off L2 regularizer)
           objective = @(w,junk) LogregL2.multinomLogregNLLGradHessL2(w, X, Y1,0,false);
           options.verbose = model.verbose; 
           if strcmpi(model.optMethod, 'projection')
             options.order = -1; % significant speed improvement with this setting
             options.maxIter = 250;
           end
           [w,output.fEvals] = L1General(model.optMethod, objective, winit,lambdaVec, options);
       end
      end % fitCore
      
    end % methods protected
    

end
