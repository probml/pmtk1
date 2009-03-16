%% Test linear Gaussian CPD with discrete parents
%#testPMTK

% C1 C2 D1 D2
%  \ | / /  \
%    Y      O
% CPD for Y is  N(y|w0 + wc'*[c1 c2] + wd'*[d1 d2], sigma^2)
% d1 has values {0,1}, d2 has values {0,1,2}

% Model
C1 = 1; C2 = 2; D1 = 3; D2 = 4; Y = 5; O=6;
G = zeros(6);
G([C1 C2 D1 D2], Y) = 1;
G(D2, O) = 1;
setSeed(0);
CPDs{C1} =  InputCPD; 
CPDs{C2} =  InputCPD;
CPDs{D1} =  TabularCPD(normalize(ones(1,2)));
CPDs{D2} =  TabularCPD(normalize(ones(1,3)));

v = [0.01 0.01];
OT = zeros(3);
AA = 1; Aa = 2; aa = 3;
OT(AA,:) = [1-v(2) v(2) 0];
OT(Aa,:) = [v(1) 1-2*v(1) v(1)];
OT(aa,:) = [0 v(2) 1-v(2)];
CPDs{O} = TabularCPD(OT);

sigma = 1;
w0 = 0.1;
wc = randn(2,1);
wd = randn(2,1);
nstates = [2 3];
cparents = [1 2];
dparents = [3 4];
child = 5;
CPDs{Y} =  LinGaussHybridCPD(w0, wc, wd,  sigma^2);
%dgm = DgmDist(G, 'CPDs', CPDs, 'infMethod', VarElimInfEng()); % works
dgm = DgmDist(G, 'CPDs', CPDs, 'infMethod', JtreeInfEng()); % works
%dgm = DgmDist(G, 'CPDs', CPDs, 'infMethod', EnumInfEng()); % cannot be used due to cts nodes

% data
xc = randn(2,1);
y = randn(1,1);

% Inference
% p(D1|C1,C2,Y)
pD1 = pmf(marginal(dgm, D1, [cparents child O], [xc; y; 2]));
% p(D1,D2|C1,C2,Y)
pD1D2 = pmf(marginal(dgm, [D1,D2], [cparents child], [xc; y]));
assert(approxeq(pD1, sum(pD1D2,2)))
