classdef LogregL2 < Logreg
%% Multinomial logistic regression with L2 prior on weights

    properties
      lambda;
    end  
    
    %% Main methods
    methods

     function m = LogregL2(varargin)
      % LogregL2(lambda, transformer, verbose,  w, w0, nclasses, optMethod,
      % labelSpace)
      % optMethod can be any minFunc method (default lbfgs)
      % or 'boundoptOverrelaxed' or 'boundoptStepwise'
      % For details on the bound optimization methods, see
      % B. Krishnapuram et al, PAMI 2004
      % "Learning sparse Bayesian classifiers: multi-class formulation, 
      %     fast algorithms, and generalization bounds"
      [m.lambda, m.transformer,  m.verbose,  m.w, m.w0, m.nclasses,  m.optMethod, m.labelSpace] = ...
        processArgs( varargin ,...
        '-lambda', [], ...
        '-transformer', [], ...
        '-verbose', false, ...
        '-w'          , [], ...
        '-w0', [], ...
        '-nclasses'   , [], ...
        '-optMethod', 'lbfgs', ...
        '-labelSpace', []);
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
       X = [ones(n,1) X];
       [n,d] = size(X);
       offsetAdded = true;
       U = unique(y);
       if isempty(model.labelSpace), model.labelSpace = U; end
       if isempty(model.nclasses), model.nclasses = length(model.labelSpace); end
       C = model.nclasses;
       Y1 = oneOfK(y, C);
       winit = zeros(d*(C-1),1);
       switch model.optMethod
         % The boundopt code regularizes w0...
         case 'boundoptOverrelaxed'
           [w, output]  = boundOptL2overrelaxed(X, Y1, model.lambda);
         case 'boundoptStepwise',
            [w, output]  = boundOptL2stepwise(X, Y1, model.lambda);
         otherwise % minFunc
           objective = @(w,junk) multinomLogregNLLGradHessL2(w, X, Y1, model.lambda,offsetAdded);
           options.Method = model.optMethod;
           options.Display = model.verbose;
           [w, f, exitflag, output] = minFunc(objective, winit, options);
       end
       W = reshape(w, d, C-1);
       model.w0 = W(1,:);
       model.w = W(2:end,:);
     end
      

    end % methods

end
