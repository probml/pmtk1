%% Inference Demo

d = 10;
G = UndirectedGraph('type', 'loop', 'nnodes', d);
ggm = UgmGaussDist(G, [], []);
ggm = mkRndParams(obj);
mvn = MvnDist(ggm.mu, ggm.Sigma);

V = 1:2; H = mysetdiff(1:d, V); xv = randn(2,1);

pggm = predict(ggm, V, xv, H);
pmvn = predict(mvn, V, xv, H);

assert(approxeq(mean(pggm), mean(pmvn)))
assert(approxeq(cov(pggm), cov(pmvn)))