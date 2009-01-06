classdef ParamJointDist < ParamDist

  % Parametric joint distribution, which supports inference
  % about states of some components (dimensions) given evidence on others
  % (Abstract class)

  properties
    infEng;
    conditioned = false;
    % conditioned =  true if infEng has been initialized
    % by calling condition
  end

  methods

    function model = condition(model, visVars, visValues)
      % enter evidence that visVars=visValues
      if nargin < 2
        visVars = []; visValues = [];
      end
      [model.infEng] = condition(model.infEng, model, visVars, visValues);
      model.conditioned = true;
    end

    function [postQuery] = marginal(model, queryVars)
      % postQuery = p(queryVars) conditional on the most recent
      % condition operation
      if ~model.conditioned
        %error('must first call condition');
        model = condition(model);
      end
      [postQuery] = marginal(model.infEng, queryVars);
    end

    function [samples] = sample(model, n)
      if ~model.conditioned, model = condition(model); end
      if nargin < 2, n = 1; end
      [samples] = sample(model.infEng,  n);
    end

    function L = logprob(model, X)
      % L(i) = log p(X(i,:) | params)
      if ~model.conditioned, model = condition(model); end
      %L = logprob(model.infEng, X, normalize);
      L = logprobUnnormalized(model, X) - lognormconst(model);
    end

    function logZ = logprobUnnormalized(model, X)
      %if ~model.conditioned, model = condition(model); end
      logZ = logprobUnnormalized(model.infEng, X);
    end
    
    function logZ = lognormconst(model)
      %if ~model.conditioned, model = condition(model); end
      logZ = lognormconst(model.infEng);
    end

    function mu = mean(model)
      if ~model.conditioned, model = condition(model); end
      mu = mean(model.infEng);
    end

    function mu = mode(model)
      if ~model.conditioned, model = condition(model); end
      mu = mode(model.infEng);
    end

    function C = cov(model)
      if ~model.conditioned, model = condition(model); end
      C = cov(model.infEng);
    end

    function C = var(model)
      if ~model.conditioned, model = condition(model); end
      C = var(model.infEng);
    end

    function Xc = impute(model, X)
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


  end

end

