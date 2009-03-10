function [wn, Sn] = normalEqnsBayes(X, y, Lam0, w0, sigma)
% numerically stable solution to posterior mean and covariance

[Lam0root, p] = chol(Lam0);
if p>0
  d=length(w0);
  Lam0root = zeros(d,d);
end
Xtilde = [X/sigma; Lam0root];
ytilde = [y/sigma; Lam0root*w0];
[Q,R] = qr(Xtilde, 0);
wn = R\(Q'*ytilde);
if nargout >= 2
  Rinv = inv(R);
  Sn = Rinv*Rinv';
end

if false % naive way, for debugging
  s2 = sigma^2;
  Sninv = Lam0 + (1/s2)*(X'*X);
  Sn2 = inv(Sninv);
  wn2 = Sn2*(Lam0*w0 + (1/s2)*X'*y);
  assert(approxeq(Sn,Sn2))
  assert(approxeq(wn,wn2))
end

end