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
        m.labelSpace, m.addOnes] = ...
        processArgs( varargin ,...
        '-lambda', [], ...
        '-transformer', [], ...
         '-verbose', false, ...
        '-w'          , [], ...
        '-w0', [], ...
        '-nclasses'   , [], ...
        '-optMethod', 'projection', ...
        '-labelSpace', [], ...
        '-addOnes', true);
     end
    
     
     function [model,output] = fit(model,D)
       % m = fit(m, D) Compute MAP estimate
       % D is DataTable containing:
       % X(i,:) is i'th input; do *not* include a column of 1s
       % y(i) is i'th response
       X = D.X; y = D.Y;
       if ~isempty(model.transformer)
         [X, model.transformer] = train(model.transformer, X);
         if addOffset(model.transformer), error('don''t add column of 1s'); end
       end
       n = size(X,1);
       if model.addOnes
         X = [ones(n,1) X];
         offsetAdded = true;
       else
         offsetAdded = false;
       end
       [n,d] = size(X);
       U = unique(y);
       if isempty(model.labelSpace), model.labelSpace = U; end
       if isempty(model.nclasses), model.nclasses = length(model.labelSpace); end
       C = model.nclasses;
       Y1 = oneOfK(y, C);
       winit = zeros(d*(C-1),1);
       switch model.optMethod
         % The boundopt code regularizes w0...
         case 'boundoptOverrelaxed'
           [w, output]  = boundOptL1overrelaxed(X, Y1, model.lambda);
         case 'boundoptStepwise',
            [w, output]  = boundOptL1stepwise(X, Y1, model.lambda);
         otherwise % minFunc
           lambdaVec = model.lambda*ones(d,C-1);
           if(offsetAdded),lambdaVec(:,1) = 0;end % don't regularize w0
           lambdaVec = lambdaVec(:);
           % unpenalized objective:
           objective = @(w,junk) multinomLogregNLLGradHessL2(w, X, Y1,0,false);
           options.verbose = model.verbose; 
           options.order = -1; % significant speed improvement with this setting
           options.maxIter = 250;
           [w,output.fEvals] = L1General(model.optMethod, objective, winit,lambdaVec, options);
       end
       W = reshape(w, d, C-1);
       if model.addOnes
         model.w0 = W(1,:);
         model.w = W(2:end,:);
       else
         model.w = W;
         model.w0 = 0;
       end
     end
      

    end % methods

end
