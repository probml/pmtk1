%% Simple Test of Discrete_DirichletDist
prior = DirichletDist(0.1*ones(3,1));
X = sampleDiscrete([0.1 0.3 0.6]', 5, 2);
m = Discrete_DirichletDist(prior);
m = fit(m, 'data', X);
v = var(m);