seed = 0; rand('state', seed); randn('state', seed);

%C = randpd(3);

n = 10; d = 3;
X = randn(n,d);
%C = (1/n)*X'*X;
C = X'*X;

[lam, u] = powerMethod(C);
%[lam2, u2] = powerMethod2(X);
%assert(approxeq(lam, lam2))
%assert(approxeq(abs(u), abs(u2)))

[uAll,lamAll] = eig(C);
[lamAll, ndx] = sort(diag(lamAll), 'descend');
u1 = uAll(:,ndx(1));
lam1 = lamAll(ndx(1));

assert(approxeq(lam, lam1))
assert(approxeq(abs(u), abs(u1)))
