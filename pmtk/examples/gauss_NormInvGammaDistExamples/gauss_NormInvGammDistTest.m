%% Simple Test of Gauss_NormInvGammaDist
prior = NormInvGammaDist('mu', 0, 'k', 0.01, 'a', 0.01, 'b', 0.01);
p = Gauss_NormInvGammaDist(prior);
x = rand(100,1);
p = fit(p, 'data', x);
pp = marginal(p);
pp = var(p);