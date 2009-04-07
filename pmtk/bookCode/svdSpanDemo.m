% based on vandenberghe p53

A = [6 0 0; 2 0 0; -4 1 -2];
[U,S,V]=svd(A,'econ'); S = diag(S);
ndx0 = find(S<=eps);
ndxPos = find(S>eps);
R = U(:,ndxPos); % range
N = V(:,ndx0); % null space

b = [3 1 10]'; % this satisfies b1=3b2
% verify that b is in the range
c = R \ b;
assert(approxeq(b, sum(R*diag(c),2)))
%approxeq(b, c(1)*U(:,1) + c(2)*U(:,2))
% compute a particular solution
z  = pinv(A)*b;
assert(approxeq(norm(A*z - b), 0))

% generate another solution
c = rand(length(ndx0),1);
x = z + sum(N*diag(c),2);
assert(approxeq(norm(A*x - b), 0))

b = [1 1 10]'; % this does not satisfies b1=3b2
% verify that b is not in the range
c = R \ b;
assert(not(approxeq(b, sum(R*diag(c),2))))
z  = pinv(A)*b;
assert(not(approxeq(norm(A*z - b), 0)))
