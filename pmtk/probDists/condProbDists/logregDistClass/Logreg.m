classdef Logreg < ProbDist
  %% Multinomial Logistic Regression
  
  properties
    w;  % d * C-1
    w0; % 1 * C-1 
    transformer;            % A data transformer object, e.g. KernelTransformer
    nclasses;               % The number of classes
    labelSpace;           % labels for y
    optMethod;
    verbose;
    addOffset; 
  end
  
  %% Main methods
  methods
    
    function m = Logreg(varargin)
       % Logreg(transformer, w, w0, nclasses, optMethod,
      % labelSpace, addOffset)
      [m.transformer, m.verbose,  m.w, m.w0, m.nclasses,  m.optMethod, ...
        m.labelSpace, m.addOffset] = ...
        processArgs( varargin ,...
        '-transformer', [], ...
        '-verbose', false, ...
        '-w'          , [], ...
        '-w0', [], ...
        '-nclasses'   , [], ...
        '-optMethod', 'lbfgs', ...
        '-labelSpace', [], ...
        '-addOffset', false);
    end
    
    
    function [model,output] = fit(model,D)
       % m = fit(m, D) Compute MLE estimate
       % D is DataTable containing:
       % X(i,:) is i'th input; do *not* include a column of 1s
       % y(i) is i'th response
       X = D.X; y = D.Y;
       if ~isempty(model.transformer)
         [X, model.transformer] = train(model.transformer, X);
         if addOffset(model.transformer), error('don''t add column of 1s'); end %#ok
       end
       n = size(X,1);
       if model.addOffset
         X = [ones(n,1) X];
       end
       d = size(X,2);
       U = unique(y);
       if isempty(model.labelSpace), model.labelSpace = U; end
       if isempty(model.nclasses), model.nclasses = length(model.labelSpace); end
       C = model.nclasses;
       Y1 = oneOfK(y, C);
       winit = zeros(d*(C-1),1);
       [W, output, model] = fitCore(model, X, Y1,  winit);
       if model.addOffset
         model.w0 = W(1,:);
         model.w = W(2:end,:);
       else
         model.w = W;
         model.w0 = 0;
       end   
    end
      
    function df = dof(model)
       % Unregularized MLE has maximal dof
       df = length(model.w(:));
    end
     
  
    
    function [yhat, pred] = predict(obj,X)
      % yhat(i) = most probable label for X(i,:)
      % pred(i) = p(y|X(i,:), w) a DiscreteDist
      if ~isempty(obj.transformer)
        X = test(obj.transformer, X);
      end
      [n,d] = size(X);
      if obj.addOffset
        X = [ones(n,1) X];
        W = [obj.w0; obj.w];
      else
        W = obj.w;
      end
      T = multiSigmoid(X,W(:)); % n*C
      pred = DiscreteDist('-T', T', '-support',obj.labelSpace);
      [p, yhat] = max(T,[],2);
      yhat = obj.labelSpace(yhat);
    end
    
    function p = logprob(obj, D)
      % p(i) = log p(y(i) | D.X(i,:), obj.w), D.y(i) in 1...C
       X = D.X; y = D.Y; 
       y = canonizeLabels(y, obj.labelSpace);
      [yhat, pred] = predict(obj,X);
      P = pmf(pred)'; % n by C
      Y = oneOfK(y, obj.nclasses);
      %p =  sum(sum(Y.*log(P)));
      p =  sum(Y.*log(P), 2);
    end
    
  end % methods
  
  methods(Access = 'protected')
    
   function [W, output, model] = fitCore(model, X, Y1,  winit) 
     tmp = LogregL2('-lambda', 0, '-optMethod', model.optMethod, ...
        '-labelSpace', model.labelSpace, '-verbose', model.verbose, ...
         '-transformer', model.transformer);
       [W, output] = fitCore(tmp, X, Y1, winit);
   end
    
  end
  
end % class