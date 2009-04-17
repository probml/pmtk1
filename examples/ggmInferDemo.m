%% Inference in GGMs
%#testPMTK
d = 10;
G = UndirectedGraph('type', 'loop', 'nnodes', d);
ggm = UgmGaussDist(G, [], []);
ggm = mkRndParams(ggm);
mvn = MvnDist(ggm.mu, ggm.Sigma);

V = 1:2; H = setdiffPMTK(1:d, V); xv = randn(2,1);

% This is a rather vacuous test since current UgmGaussDist
% actually uses MvnDist's inference method

pggm = marginal(ggm, H, V, xv);
pmvn = marginal(mvn, H, V, xv);

assert(approxeq(mean(pggm), mean(pmvn)))
assert(approxeq(cov(pggm), cov(pmvn)))