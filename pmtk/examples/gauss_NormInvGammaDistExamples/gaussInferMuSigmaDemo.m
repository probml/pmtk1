%% Demo of inferring mu and sigma for a 1d Gaussian
% Compare to mvnInferMuSigma1d in Mvn_MvnInvWishartDistExamples
setSeed(1);
muTrue = 5; varTrue = 10;
X = sample(GaussDist(muTrue, varTrue), 500);
muRange = [0 10]; sigmaRange  = [0.1 12];
figure; hold on;
[styles, colors, symbols] =  plotColors();
ns = [0 2 5 50 500]
probs = {};
h = zeros(length(ns),1);
for i=1:length(ns)
    prior2 = NormInvGammaDist('mu', 0, 'k', 0.001, 'a', 0.001, 'b', 0.001);
    n = ns(i);
    m2 = fit(Gauss_NormInvGammaDist(prior2), 'data', X(1:n));
    post2 = m2.muSigmaDist;
    plot(post2, 'plotArgs', {styles{i}, 'linewidth', 2}, ...
        'xrange', [muRange sigmaRange], 'useContour', true);
    legendstr{i} = sprintf('n=%d', n);
    xbar = mean(X(1:n)); vbar = var(X(1:n));
    h(i)=plot(xbar, vbar, 'x','color',colors(i),'markersize', 12,'linewidth',3);
end
xlabel(sprintf('%s', '\mu'))
ylabel(sprintf('%s', '\sigma^2'))
legend(h,legendstr);
title('NIG posteriors')
  