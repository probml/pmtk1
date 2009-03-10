
% Test linear Gaussian CPD with discrete parents
% C1 C2 D1 D2
%  \ | / /
%    Y
% factor(d1, d2) = N(y|w0 + wc'*[c1 c2] + wd'*[d1 d2], sigma^2)

% d1 has values {0,1}, d2 has values {0,1,2}
setSeed(0);
sigma = 1;
w0 = 0.1;
wc = randn(2,1);
wd = randn(2,1);
xc = randn(2,1);
y = randn(1,1);
nstates = [2 3];
ndx = 1;
prob = [];
for d2=1:nstates(2) % toggle d1 faster
  for d1=1:nstates(1)
    mu = w0 + wc'*xc + wd'*([d1 d2]'-1);
    prob(ndx) = normpdf(y, mu, sigma);
    ndx = ndx + 1;
  end
end

cparents = [1 2];
dparents = [3 4];
child = 5;
CPD = LinGaussHybridCPD(w0, wc, wd,  sigma^2, dparents, nstates);
Tfac = convertToTabularFactor(CPD,  1:5, [cparents child], [xc; y]);
approxeq(prob(:), Tfac.T(:))
