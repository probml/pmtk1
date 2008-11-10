classdef ProbDist
  % probability density function
  
  properties
  end

  %{
  methods(Abstract = true)
% Methods implemented by most child classes
    h = plot(obj, varargin); % h is a figure handle (or cell array)
    m = mean(obj); % [E[X1],...,E[Xd]] (constant vector), not the scalar RV 1/d sum_j X_j
    m = mode(obj);
    v = var(obj);
    X = sample(obj, n); % X(i,j) = sample from params(j), an nxd matrix
    p = logprob(obj, D); % p(i,j) = log p(D(i) | params(j))
    logZ = lognormconst(obj);
    obj = fit(obj, varargin);
    pr = predict(obj, D); 
  end
  %}
  %%  Main methods
  methods
   
    %{
    function nll = negloglik(obj, X)
      % Negative log likelihood of a data set
      % For scalar dist: nll = -sum_i log p(X(i) | params)
      % For vector distr: nll = -sum_i log p(X(i,:) | params)
      % For matrix distr: nll = -sum_i log p(X(:,:,i) | params)
       % For factorized scalar dist: nll(j) = -sum_i log p(X(i) | params(j))
       nll = -sum(logprob(obj, X),1);
    end
     %}
    
    function d = nfeatures(obj)
      % num dimensions (variables)
      mu = mean(obj); d = length(mu);
    end
    
    function [mu, stdErr] = cvScore(obj, X, varargin)
      [nfolds] = process_options(varargin, 'nfolds', 5);
      [n d] = size(X);
      [trainfolds, testfolds] = Kfold(n, nfolds);
      score = zeros(1,n);
      for k = 1:nfolds
        trainidx = trainfolds{k}; testidx = testfolds{k};
        Xtest = X(testidx,:);  Xtrain = X(trainidx, :); 
        obj = fit(obj, 'data', Xtrain);
        score(testidx) = logprob(obj,  Xtest);
        %fprintf('fold %d, logprob %5.3f, mu %5.3f, sigma %5.3f\n', k, L(k), obj.mu, obj.sigma2);
      end
      mu = mean(score);
      stdErr = std(score,0,2)/sqrt(n);
    end
    
  end
  
 
end