classdef LogregBinary < ProbDist
  %% Binary Logistic Regression
  
  properties
    w;  % d * 1
    w0; % 1 * 1 
    transformer;            % A data transformer object, e.g. KernelTransformer
    labelSpace;           % y is [0,1] or [-1,1] or [1,2]
    optMethod;
    verbose;
    addOffset;
  end
  
  %% Main methods
  methods
    
    function m = LogregBinary(varargin)
       % LogregBinary(transformer, verbose, w, w0, optMethod,
      % labelSpace)
      [m.transformer, m.verbose,  m.w, m.w0,  m.optMethod, m.labelSpace, ...
        m.addOffset] = ...
        processArgs( varargin ,...
        '-transformer', [], ...
        '-verbose', false, ...
        '-w'          , [], ...
        '-w0', [], ...
        '-optMethod', 'lbfgs', ...
        '-labelSpace', [], ...
        '-addOffset', true);
    end
    
    function model = fit(model,D)
      % m = fit(m, D) Compute MLE (not recommended)
      % D is DataTable containing:
      % X(i,:) is i'th input; do *not* include a column of 1s
      % y(i) is i'th response
      tmp = LogregBinaryL2('-lambda', 0, '-optMethod', model.optMethod, ...
        '-labelSpace', model.labelSpace, '-verbose', model.verbose, ...
        '-addOffset', model.addOffset);
      tmp = fit(tmp, D);
      model.w = tmp.w; model.w0 = tmp.w0;
      model.labelSpace = tmp.labelSpace;
    end
    
    
    function [yhat, pred] = predict(obj,X)
      % yhat(i) = most probable label for X(i,:)
      % pred(i) = p(y|X(i,:), w) a BernoulliDist
      if ~isempty(obj.transformer)
        X = test(obj.transformer, X);
      end
      [n,d] = size(X);
      X = [ones(n,1) X];
      w = [obj.w0; obj.w];
      p = sigmoid(X*w); % p<0.5, y=2, p>0.5, y=1
      yhat = ones(n,1);
      ndx2 = (p > 0.5);
      yhat(ndx2) = 2;
      yhat = obj.labelSpace(yhat);
      if nargout >= 2
        pred = BernoulliDist('-mu', p, '-support',obj.labelSpace);
      end
    end
    
    function p = logprob(model, D)
      % p(i) = log p(y(i) | D.X(i,:), obj.w)
      % D.y(i) in obj.support 
      X = D.X; y = D.Y;
      y = canonizeLabels(y, model.labelSpace);
      [yhat, pred] = predict(model, X);
      p1 = pmf(pred); % n by 1
      p = (y==1)*log(p1) + (y==2)*log(1-p1);
    end
    
  end % methods
  
  
  
end % class