%% Inference Demo
d = 10;
G = UndirectedGraph('type', 'loop', 'nnodes', d);
obj = GgmDist(G, [], []);
obj = mkRndParams(obj);
V = 1:2; H = mysetdiff(1:d, V); xv = randn(2,1);
obj = enterEvidence(obj, V, xv);
pobj = marginal(obj, H);
obj2 = MvnDist(obj.mu, obj.Sigma);
obj2 = enterEvidence(obj2, V, xv);
pobj2 = marginal(obj2, H);
assert(approxeq(mean(pobj), mean(pobj2)))
assert(approxeq(cov(pobj), cov(pobj2)))