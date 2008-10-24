function mu = updateMean(mu, SS, Sigma); 

S0 = mu.Sigma; S0inv = inv(S0);
mu0 = mu.mu;
S = Sigma; Sinv = inv(S);
n = SS.n;
Sn = inv(inv(S0) + n*Sinv);
mu = mvnDist(Sn*(n*Sinv*SS.xbar + S0inv*mu0), Sn);
