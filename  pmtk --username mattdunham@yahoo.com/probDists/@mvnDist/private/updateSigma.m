function Sigma = updateSigma(Sigma, mu, SS)

n = SS.n;
T0 = Sigma.Sigma;
v0 = Sigma.dof;
vn = v0 + n;
Tn = T0 + n*SS.XX +  n*(SS.xbar-mu)*(SS.xbar-mu)';
Sigma = invWishartDist(vn, Tn);