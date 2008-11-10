function w = updateMVNIG(obj, X, y);
a0 = obj.w.a; b0 = obj.w.b; w0 = obj.w.mu; S0 = obj.w.Sigma;
v0 = 2*a0; s02 = 2*b0/v0;
d = length(w0);
if det(S0)==0
  noninformative = true;
  Lam0 = zeros(d,d);
else
  noninformative = false;
  Lam0 = inv(S0);
end
[wn, Sn] = normalEqnsBayes(X, y, Lam0, w0, 1);
n = size(X,1);
vn = v0 + n;
an = vn/2;
if noninformative
  sn2 = (1/vn)*(v0*s02 + (y-X*wn)'*(y-X*wn));
else
  sn2 = (1/vn)*(v0*s02 + (y-X*wn)'*(y-X*wn) + (wn-w0)'*Sn*(wn-w0));
end
bn = vn*sn2/2;
w = MvnInvGammaDist('mu', wn, 'Sigma', Sn, 'a', an, 'b', bn);
end