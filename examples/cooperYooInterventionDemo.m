%% Verify Marginal Likelihood Equation with Interventional Data
% "Causal Discovery from a Mixture of Experimental and
% Observational Data" Cooper & Yoo, UAI 99, sec 2.2
%#testPMTK
G = zeros(2,2); G(1,2) = 1;
CPD{1} = TabularCPD([0.5 0.5], 'prior', 'BDeu');
CPD{2} = TabularCPD(mkStochastic(ones(2,2)), 'prior', 'BDeu');
dgm = DgmDist(G, 'CPDs', CPD);
X = [2 2; 2 1; 2 2; 1 1; 1 2; 2 2; 1 1; 2 2; 1 2; 2 1; 1 1];
M = [0 0; 0 0; 0 0; 0 0; 0 0; 1 0; 1 0; 0 1; 0 1; 0 1; 0 1];
M = logical(M);
dgm = fit(dgm, X, 'interventionMask', M);
L = exp(sum(logprob(dgm, X, 'interventionMask', M))) % plugin
L = exp(logmarglik(dgm, X, 'interventionMask', M)) % Bayes
assert(approxeq(L, 5.97e-7))