%% Demo of inferring mu and sigma for a 1d Gaussian
clear;
seed = 0; rand('twister', seed); randn('state', seed);
muTrue = 10; varTrue = 5^2;
N = 12;
X = sample(MvnDist(muTrue, varTrue), N);
%X = normrnd(muTrue, sqrt(varTrue), N, 1);
%X = [141, 102, 73, 171, 137, 91, 81, 157, 146, 69, 121, 134];
v = 1; S = var(X);
prior{1} = MvnInvWishartDist('mu', mean(X), 'k', 1, 'dof', v, 'Sigma', v*S);
names{1} = 'Data-driven'; % since has access to data
v = 0; S = 0;
prior{2} = MvnInvWishartDist('mu', 0, 'k', 0.01, 'dof', v, 'Sigma', v*S);
names{2} = 'Jeffreys'; % Jeffrey
v = N; S = 10;
prior{3} = MvnInvWishartDist('mu', 5, 'k', N, 'dof', v, 'Sigma', v*S);
names{3} = 'Wrong';
muRange = [0 20]; sigmaRange  = [1 30];
nr = 3; nc = 3;
figure;
for i=1:3
    m = fit(MvnDist(prior{i}, []), 'data', X);
    post{i} = m.mu;
    pmuPost = marginal(post{i}, 'mu');
    pSigmaPost = marginal(post{i}, 'Sigma');
    pmuPrior = marginal(prior{i}, 'mu');
    pSigmaPrior = marginal(prior{i}, 'Sigma');

    subplot2(nr,nc,i,1);
    plot(pmuPrior, 'plotArgs', {'k:', 'linewidth',2}, 'xrange', muRange); hold on
    plot(pmuPost, 'plotArgs', {'r-', 'linewidth', 2}, 'xrange', muRange);
    title(sprintf('p(%s|D) %s', '\mu', names{i}(1)))
    %legend('prior', 'post')

    subplot2(nr,nc,i,2);
    plot(pSigmaPrior, 'plotArgs', {'k:', 'linewidth',2}, 'xrange', sigmaRange); hold on
    plot(pSigmaPost, 'plotArgs', {'r-', 'linewidth', 2}, 'xrange', sigmaRange);
    title(sprintf('p(%s|D) %s', '\sigma^2', names{i}(1)))

    subplot2(nr,nc,i,3);
    plot(prior{i}, 'plotArgs', {'k:', 'linewidth',2}, ...
        'xrange', [muRange sigmaRange], 'useContour', true); hold on
    plot(post{i}, 'plotArgs', {'r-', 'linewidth', 2}, ...
        'xrange', [muRange sigmaRange], 'useContour', true);
    %title(sprintf('p(%s,%s|D) %s', '\mu', '\sigma^2', names{i}));
    title(sprintf('%s', names{i}));
    %xlabel(sprintf('%s','\mu')); ylabel(sprintf('%s','\sigma^2'));
end