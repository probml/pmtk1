%% Illustrate sampling from a GMM using a Gaussian proposal
%#author Christoph Andrieu

sigmas = [10 100 500];
for i=1:length(sigmas)

    sigma_prop = sigmas(i);
    setSeed(0);
    m = MixGaussDist('K', 2, 'mu', [-50 50], 'Sigma', reshape([10^2 10^2], [1 1 2]), ...
        'mixweights', [0.3 0.7]);
    eng = McmcInfer('Nsamples', 1000, 'Nburnin', 0, 'method', 'metrop');
    eng.targetFn = @(x) (logprob(m,x));
    eng.proposalFn = @(x) (x + (sigma_prop * randn(1,1)));
    eng.xinitFn = @() (m.mu(2));
    eng = enterEvidence(eng, [], []);
    x = eng.samples;

    figure;
    nb_iter = eng.Nsamples;
    x_real = linspace(-100, 100, nb_iter);
    y_real = exp(logprob(m, x_real(:)));
    Nbins = 100;
    plot3(1:nb_iter, x, zeros(nb_iter, 1))
    hold on
    plot3(ones(nb_iter, 1), x_real, y_real)
    [u,v] = hist(x, linspace(-100, 100, Nbins));
    plot3(zeros(Nbins, 1), v, u/nb_iter*Nbins/200, 'r')
    hold off
    grid
    view(60, 60)
    xlabel('Iterations')
    ylabel('Samples')
    title(sprintf('MH with N(0,%5.3f^2) proposal', sigma_prop))


end



