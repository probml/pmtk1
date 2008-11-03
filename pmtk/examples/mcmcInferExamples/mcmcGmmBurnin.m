%% GMM Burnin Example
seeds = 1:3;
sigmas = [10 100 500];
%sigmas = [10];
N = 1000;
X = zeros(N,  length(seeds), length(sigmas));
for s=1:length(sigmas)
    m = gaussMixDist('K', 2, 'mu', [-50 50], 'Sigma', reshape([10^2 10^2], [1 1 2]), ...
        'mixweights', [0.3 0.7]);
    eng = mcmcInfer('Nsamples', N, 'Nburnin', 0, 'method', 'metrop', 'seeds', seeds);
    eng.targetFn = @(x) (logprob(m,x));
    eng.proposalFn = @(x) (x + (sigmas(s) * randn(1,1)));
    eng.xinitFn = @() (m.mu(2));
    eng = enterEvidence(eng, [], []);
    X(:,:,s) = eng.samples;
end
colors = {'r', 'g', 'b', 'k'};
% Trace plots
for s=1:length(sigmas)
    figure; hold on;
    for i=1:length(seeds)
        plot(X(:,i,s), colors{i});
    end
    Rhat(s) = epsr(X(:,:,s));
    title(sprintf('sigma prop = %5.3f, Rhat = %5.3f', sigmas(s), Rhat(s)))
end
% Smoothed trace plots
for s=1:length(sigmas)
    figure; hold on
    for i=1:length(seeds)
        movavg = filter(repmat(1/50,50,1), 1, X(:,i,s));
        plot(movavg,  colors{i});
    end
    title(sprintf('sigma prop = %5.3f, Rhat = %5.3f', sigmas(s), Rhat(s)))
end
% Plot auto correlation function for 1 chain
for s=1:length(sigmas)
    figure;
    stem(acf(X(:,1,s), 20));
    title(sprintf('sigma prop = %5.3f', sigmas(s)))
end
