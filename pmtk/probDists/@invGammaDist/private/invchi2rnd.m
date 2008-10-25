function xs = invchi2rnd(v, s2, m, n)
% Draw an m*n matrix of inverse chi squared RVs, v = dof, s2=scale
% Gelman p580
xs = v*s2./chi2rnd(v, m, n);


