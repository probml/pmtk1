function M = randpd(d)
% Create a random positive definite matrix of size d by d 

A = randn(d);
M = A*A';
assert(isposdef(M))
