classdef Logreg < ParamDist
  %% Multinomial Logistic Regression
  
  properties
    w;  % d * C-1
    w0; % 1 * C-1 
    transformer;            % A data transformer object, e.g. KernelTransformer
    nclasses;               % The number of classes
    labelSpace;           % labels for y
    optMethod;
    verbose;
  end
  
  %% Main methods
  methods
    
    function m = Logreg(varargin)
       % Logreg(transformer, w, w0, nclasses, optMethod,
      % labelSpace)
      [m.transformer, m.verbose,  m.w, m.w0, m.nclasses,  m.optMethod, m.labelSpace] = ...
        processArgs( varargin ,...
        '-transformer', [], ...
        '-verbose', false, ...
        '-w'          , [], ...
        '-w0', [], ...
        '-nclasses'   , [], ...
        '-optMethod', 'lbfgs', ...
        '-labelSpace', []);
    end
    
    function model = fit(model,D)
      % m = fit(m, D) Compute MLE (not recommended)
      % D is DataTable containing:
      % X(i,:) is i'th input; do *not* include a column of 1s
      % y(i) is i'th response
      tmp = LogregL2('-lambda', 0, '-optMethod', model.optMethod, ...
        '-labelSpace', model.labelSpace, '-verbose', model.verbose, ...
         '-transformer', model.transformer);
      tmp = fit(tmp, D);
      model.w = tmp.w; model.w0 = tmp.w0;
      model.labelSpace = tmp.labelSpace;
      model.transformer = tmp.transformer;
    end
    
    
    function [yhat, pred] = predict(obj,X)
      % yhat(i) = most probable label for X(i,:)
      % pred(i) = p(y|X(i,:), w) a DiscreteDist
      if ~isempty(obj.transformer)
        X = test(obj.transformer, X);
      end
      [n,d] = size(X);
      X = [ones(n,1) X];
      W = [obj.w0; obj.w];
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
  
  
  
end % class