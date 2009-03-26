%% (Full) Gibbs sampling from a mixture of multivariate normals
%#testPMTK
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
  logprobtrue = logprobtrue + sum(logprob(MvnDist(mu(j,:), Sigma(:,:,j)), X(p == j,:)));
end

% specify the prior distribution to use.
chosenPrior = MvnInvWishartDist('mu', mean(X), 'Sigma', diag(var((X))), 'dof', d + 1, 'k', 0.001);
model = MvnMixDist('distributions',copy( MvnDist(zeros(d,1),diag(ones(d,1)), 'prior', chosenPrior), K,1) ) ;

% Set the prior distribution on the mixing weights to be Dirichlet(1,..., 1)
model.mixingWeights.prior = DirichletDist(ones(K,1));

% Initiate the sampler
mcmc = latentGibbsSample(model,X);

% Post-process - will add in later
%[permOut] = processLabelSwitch(model,mcmc,X);

% Plot some interesting results - point estimates
plot(mcmc.loglik, 'linewidth', 3);
xlabel('Iteration (after accounting for burnin and thinning)');
ylabel('log likelihood');
title('log likelihood vs iteration for the Gibbs Sampler');

adjrand = zeros(length(mcmc.loglik),1);
rand = zeros(length(mcmc.loglik),1);
for j=1:length(mcmc.loglik)
[adjrand(j),rand(j),junk1,junk2] = valid_RandIndex(mcmc.latent(:,j),p);
end

bestrand = mcmc.latent(:,argmax(rand));
bestll = mcmc.latent(:,argmax(mcmc.loglik));

colortrue = {'bo', 'go', 'ro', 'co', 'mo'};
colorrand = {'bx', 'gx', 'rx', 'cx', 'mx'};

figure(); hold on;
for j=1:K
    plot(X(p == j,1), X(p == j,2), colortrue{j}, 'linewidth', 3, 'MarkerSize', 10);
    plot(X(bestrand == j,1), X(bestrand == j,2), colorrand{j}, 'linewidth', 3, 'MarkerSize', 10);
end
title('Real vs. Gibbs clustering for iteration achieving best rand index.')

figure(); hold on;
for j=1:K
    plot(X(p == j,1), X(p == j,2), colortrue{j}, 'linewidth', 3, 'MarkerSize', 10);
    plot(X(bestll == j,1), X(bestll == j,2), colorrand{j}, 'linewidth', 3, 'MarkerSize', 10);
end
title('Real vs. Gibbs clustering for iteration achieving best log likelihood')
