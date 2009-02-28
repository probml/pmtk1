%% Inference in undirected version of sprinkler network
% Compare to sprinklerDGMdemo
%#testPMTK

dgm = mkSprinklerDgm();
ugm = convertToUgm(dgm);
ugm.infMethod = 'enum';
model = ugm;
false = 1; true = 2;
C = 1; S = 2; R = 3; W = 4;
mW = marginal(model, W);
assert(approxeq(mW.T(true), 0.6471))

mW = marginal(model, [S, W]);
assert(approxeq(mW.T(true,true), 0.2781))

pSgivenW = pmf(conditional(model, W, true, S));
assert(approxeq(pSgivenW(true), 0.4298));
pSgivenWR = pmf(conditional(model, [W R], [true, true], S));
assert(approxeq(pSgivenWR(true), 0.1945)); % explaining away

