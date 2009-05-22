classdef LogregBinaryMc < ProbDist 
%% Binary logistic regression with Gaussian prior
% We use Monte Carlo approximation to the posterior

properties
  paramDist; % bag of samples
  transformer;            % A data transformer object, e.g. KernelTransfor
  lambda; % precision of diagonal Gaussian
  fitEng;
  addOffset;
end




%% Main methods
methods
  
  function m = LogregBinaryMc(varargin)
    % LogregBinaryMc(transformer, fitEng, addOffset)
    [m.lambda, m.transformer, m.fitEng,  m.addOffset] = ...
      processArgs( varargin ,...
      '-lambda', [], ...
      '-transformer', [], ...
      '-fitEng', LogregBinaryMhFitEng(),...
      '-addOffset', true);
  end
  
  function [model,output] = fit(model,D)
    % m = fit(m, D) Compute posterior estimate
    % D is DataTable containing:
    % X(i,:) is i'th input; do *not* include a column of 1s
    % y(i) is i'th response
    [model, output]  = fit(model.fitEng, model, D);
  end
  
  
  function pw = getParamPost(model) % SampleDist
    pw = SampleDist(model.paramDist.wsamples, model.paramDist.weights);
  end
  
  function [yhat, pred] = predict(model,D)
    % yhat(i) = most probable label for X(i,:)
    % pred(i,s) = p(y|X(i,:), beta(s)) a SampleDist
    X = D.X;
    if ~isempty(model.transformer)
      X = test(model.transformer, X);
    end
    if model.addOffset
      n = size(X,1);
      X = [ones(n,1) X];
    end
    wsamples = model.paramDist.wsamples; % wsamples(j,s), j=1 for w0
    psamples = sigmoid(X*wsamples);
    pred = SampleDist(psamples, model.paramDist.weights);
    p = mean(pred); 
    yhat = zeros(n,1);
    ndx2 = (p > 0.5);
    yhat(ndx2) = 1;
    %yhat = convertLabelsToUserFormat(D, yhat, '01'); Why do we need to do this in this case?  It is already '01'
  end
  
  function p = logprob(obj, D, method)
    % p(i) = log sum_s p(y(i) | D.X(i,:), beta(s)) w(s)
    % where w(s) is weight for sample s
    if nargin < 3, method = 1; end
    X = D.X; y = getLabels(D, '01');
    n = size(X,1);
    [yhat, pred] = predict(obj,D); % pred is n*S
    p1 = pred.samples;
    p0 = 1-p1;
    nS = size(p1,2);
    W = repmat(pred.weights, n, 1);
    mask1 = repmat((y==1), 1, nS);
    mask0 = repmat((y==0), 1, nS);
    pp = (mask1 .* p1 + mask0 .* p0) .* W;
    p1 = log(sum(pp,2));
   
    %plug in posterior mean
    mu1 = mean(pred); 
    Y = oneOfK(y, 2);
    P = [1-mu1(:) mu1(:)];
    p2 =  sum(Y.*log(P), 2);
    
    if method==1
      p = p1;
    else
      p = p2;
    end
 
  end
  
end % methods


end

