%% Simple Test of the DiscreteDist Class
%#testPMTK
p=DiscreteDist('T', [0.3 0.2 0.5]', 'support', [-1 0 1]);
X=sample(p,1000);
logp = logprob(p,[0;1;1;1;0;1;-1;0;-1]);
nll  = negloglik(p,[0;1;1;1;0;1;-1;0;-1]);
%p = fit(p, 'data', X, 'prior', 'dirichlet', 'priorStrength', 2);
p.prior = 'dirichlet'; p.priorStrength = 2;
p = fit(p, 'data', X);