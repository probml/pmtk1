function mu = updateMeanCov(mu, SS)
k0 = mu.k; m0 = mu.mu; T0 = mu.Sigma; v0 = mu.dof;
n = SS.n;
kn = k0 + n;
vn = v0 + n;
Tn = T0 + n*SS.XX + (k0*n)/(k0+n)*(SS.xbar-m0)*(SS.xbar-m0)';
mn = (k0*m0 + n*SS.xbar)/kn;
mu = mvnInvWishartDist('mu',mn, 'Sigma', Tn, 'dof', vn, 'k', kn);
