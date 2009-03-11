%% Test linear Gaussian CPD with discrete parents
%#testPMTK

% C1 C2 D1 D2
%  \ | / /
%    Y
% CPD for Y is  N(y|w0 + wc'*[c1 c2] + wd'*[d1 d2], sigma^2)
% d1 has values {0,1}, d2 has values {0,1,2}

% Model
C1 = 1; C2 = 2; D1 = 3; D2 = 4; Y = 5;
G = zeros(5,5);
G([C1 C2 D1 D2], Y) = 1;
setSeed(0);
CPDs{C1} =  InputCPD; % LinGaussCPD([], 0, 1);
CPDs{C2} =  InputCPD;
CPDs{D1} =  TabularCPD(normalize(ones(1,2)));
CPDs{D2} =  TabularCPD(normalize(ones(1,3)));

sigma = 1;
w0 = 0.1;
wc = randn(2,1);
wd = randn(2,1);
nstates = [2 3];
cparents = [1 2];
dparents = [3 4];
child = 5;
CPDs{Y} =  LinGaussHybridCPD(w0, wc, wd,  sigma^2, dparents, nstates);
dgm = DgmDist(G, 'CPDs', CPDs, 'infEng', VarElimInfEng());
%dgm = DgmDist(G, 'CPDs', CPDs, 'infEng', JtreeInfEng()); % broken
%dgm = DgmDist(G, 'CPDs', CPDs, 'infEng', EnumInfEng()); % cannot be used due to cts nodes

% data
xc = randn(2,1);
y = randn(1,1);

% Inference
% p(D1|C1,C2,Y)
pD1 = pmf(marginal(dgm, D1, [cparents child], [xc; y]));
% p(D1,D2|C1,C2,Y)
pD1D2 = pmf(marginal(dgm, [D1,D2], [cparents child], [xc; y]));
assert(approxeq(pD1, sum(pD1D2,2)))

%{
ndx = 1;
prob = [];
for d2=1:nstates(2) % toggle d1 faster
  for d1=1:nstates(1)
    mu = w0 + wc'*xc + wd'*([d1 d2]'-1);
    prob(ndx) = normpdf(y, mu, sigma);
    ndx = ndx + 1;
  end
end
CPD = CPDs{Y};
Tfac = convertToTabularFactor(CPD,  1:5, [cparents child], [xc; y]);
approxeq(prob(:), Tfac.T(:))
%}
