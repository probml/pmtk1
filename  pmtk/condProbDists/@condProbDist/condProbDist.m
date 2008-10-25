classdef condProbDist
  % conditional probability density function (y|x)
  
  properties
    ndimsX;
    ndimsY = 1;
  end

  %%  Main methods
  methods
    
    function [mu, stdErr] = cvScore(obj, X, Y, varargin)
      % X(i,:), Y(i,:)
      [nfolds, objective, randOrder] = process_options(...
        varargin, 'nfolds', 5, 'objective', 'logprob','randOrder', true);
      [n d] = size(X);
      [trainfolds, testfolds] = Kfold(n, nfolds, randOrder);
      score = zeros(1,n);
      for k = 1:length(trainfolds)
        trainidx = trainfolds{k}; testidx = testfolds{k};
        Xtest = X(testidx,:);  Xtrain = X(trainidx, :); 
        Ytest = Y(testidx,:);  Ytrain = Y(trainidx, :);
        obj = fit(obj, 'X', Xtrain, 'y', Ytrain);
        switch lower(objective)
          case 'logprob'
            score(testidx) = logprob(obj,  Xtest, Ytest);
          case 'squarederr'
            score(testidx) = squaredErr(obj,  Xtest, Ytest);
          otherwise
            error(['unrecognized objective ' objective]);
        end
        %fprintf('fold %d, logprob %5.3f\n', k, L(k));
      end
      if 0
        obj = fit(obj, 'X', X, 'y', Y);
        scoreAll = squaredErr(obj,  X, Y);
        figure;
        plot(score); hold on; 
        h= line([1 n], [scoreAll scoreAll]); set(h,'color','r');
      end
      mu = mean(score);
      stdErr = std(score, 0, 2)/sqrt(n);
    end
    

    
  end

end