%% Simple Test of Logreg_MvnDist
n = 10; d = 3; C = 2;
X = randn(n,d );
y = sampleDiscrete((1/C)*ones(1,C), n, 1);
mL2 = Logreg_MvnDist('nclasses', C, 'priorStrength', 1);
mL2 = fit(mL2, 'X', X, 'y', y, 'infMethod', 'laplace');
pred1 = predict(mL2, X, 'method', 'integral');
pred2 = predict(mL2, X, 'method', 'mc');
mL3 = fit(mL2, 'X', X, 'y', y, 'infMethod', 'mh');
pred3 = predict(mL2, X);
llL2 = logprob(mL2, X, y);