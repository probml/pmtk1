function [post,  yhat] = gaussianClassifierApply(Xtest, params, useNB, useBayes);
% gaussianClassifierApply - apply Bayes rule with Gaussian class-conditioanl densities.
% Computes post(i,c) = P(C=c|x(i,:)) u
% and yhat(i) = arg max_c post(i,c)

if nargin < 3, useNB = 1; end
if nargin < 4, useBayes = 0; end

[N d] = size(Xtest);
Nclasses = length(params.classPrior);
loglik = zeros(N, Nclasses);
for c=1:Nclasses
  if useBayes
    tmp = zeros(N,d);
    for j=1:d
      m = params.post.mu(j,c);
      k = params.post.kappa(j,c);
      a = params.post.alpha(j,c);
      b = params.post.beta(j,c);
      s2 = b*(k+1)/(a*k);
      tmp(:,j) = log(studentTpdf(Xtest(:,j), 2*a, m, s2));
    end
    loglik(:,c) = sum(tmp,2); % naive Bayes assumption
  else
    if useNB
      tmp = zeros(N,d);
      for j = 1:d
	tmp(:,j) = log(normpdf(Xtest(:,j), params.mu(j,c), params.sigma(j,c)) + eps);
      end
      loglik(:,c) = sum(tmp,2); % naive Bayes assumption
    else
      %lik(:,c) = mvnpdf(Xtest, params.mu(:, c)', params.Sigma(:,:, c));
      loglik(:,c) = log(gausspdf(Xtest, params.mu(:, c)', params.Sigma(:,:, c)));
    end
  end
end
classPrior = params.classPrior;
N = size(Xtest,1);
logjoint = loglik + repmat(log(classPrior(:)'), N, 1);
logpost = logjoint - repmat(logsumexp(logjoint,2), 1, Nclasses);
post = exp(logpost);
[junk, yhat] = max(post,[],2);
