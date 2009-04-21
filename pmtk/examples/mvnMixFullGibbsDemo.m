%% (Full) Gibbs sampling from a mixture of multivariate normals
%#author Cody Severinski

setSeed(0);
% Set the number of clusters K
% the number of observations n to generate in d dimensions,
% and the true mu, Sigma for these
K = 5; n = 500; d = 2;
mu = sample(MvnDist(zeros(d,1), 5*eye(d)), K);
Sigma = sample(InvWishartDist(d + 1, eye(d)), K);
p = unidrnd(K,n,1);

X = mvnrnd(mu(p,:),Sigma(:,:,p));

% Determine the true log probability of the data
logprobtrue = 0;
for j=1:K
  logprobtrue = logprobtrue + log(sum(p==k) / n) + sum(logprob(MvnDist(mu(j,:), Sigma(:,:,j)), X(p == j,:)));
end

% specify the prior distribution to use.
chosenPrior = MvnInvWishartDist('mu', mean(X), 'Sigma', diag(var((X))), 'dof', d + 1, 'k', 0.001);
model = MixMvnGibbs('distributions',copy( MvnDist(zeros(d,1),diag(ones(d,1)), '-prior', chosenPrior), K,1) ) ;

% Initiate the sampler
dists = latentGibbsSample(model,X, 'verbose', true);

N = size(dists.latentDist.samples,1);
latent = dists.latentDist.samples;
mix = dists.mixDist.samples;
musample = dists.muDist.samples;
Sigmatmp = dists.SigmaDist.samples;
Sigmasample = zeros(d,d,N,K);

for s=1:N
  for k=1:K
    w = Sigmatmp(s,:,k);
    Sigmasample(:,:,s,k) = reshape(w',d,d)'*reshape(w',d,d);
  end
end

loglik = zeros(N,1);

for s=1:N
  for k=1:K
    model.distributions{k}.mu = musample(s,:,k);
    model.distributions{k}.Sigma = Sigmasample(:,:,s,k);
    loglik(s) = loglik(s) + log(mix(s,k)) + sum(logprob(model.distributions{k}, X(latent(s,:) == k,:)) );
  end
end

colortrue = {'bo', 'go', 'ro', 'co', 'mo'};
colorestm = {'bx', 'gx', 'rx', 'cx', 'mx'};

bestll = argmax(loglik);
figure(); hold on;
plot(loglik, 'linewidth', 3);
line([1,N],[logprobtrue, logprobtrue], 'color', 'green', 'linewidth', 3);
plot(bestll, loglik(bestll), 'ro', 'MarkerSize', 10);
legend('MCMC log-liklihood', 'true log-likelihood', 'MAP estimate');

figure(); hold on;
for k=1:K
    plot(X(p == k,1), X(p == k,2), colortrue{k}, 'linewidth', 2, 'MarkerSize', 10);
    plot(X(p == k,1), X(p == k,2), colorestm{k}, 'linewidth', 2, 'MarkerSize', 10);
end
title('Real vs. Gibbs clustering for iteration achieving best log likelihood')
