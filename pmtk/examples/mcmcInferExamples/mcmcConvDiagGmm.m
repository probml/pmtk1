%% MCMC for Mixture of Gaussians
m = gaussMixDist('K', 2, 'mu', [-50 50], 'Sigma', reshape([10^2 10^2], [1 1 2]), ...
    'mixweights', [0.3 0.7]);
seeds = 1:3;
sigmas = [10 100 500];
N = 1000;
for s=1:length(sigmas)
    eng = mcmcInfer('Nsamples', N, 'Nburnin', 0, 'method', 'metrop', 'seeds', seeds);
    eng.targetFn = @(x) (logprob(m,x));
    eng.proposalFn = @(x) (x + (sigmas(s) * randn(1,1)));
    eng.xinitFn = @() (m.mu(2));
    eng = enterEvidence(eng, [], []);
    X = eng.samples;
    mcmcInfer.plotConvDiagnostics(X, 1, sprintf('sigma prop %5.3f', sigmas(s)));
end