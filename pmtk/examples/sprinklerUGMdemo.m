%% Inference in undirected version of sprinkler network
% Compare to sprinklerDGMdemo
%#testPMTK

dgm = mkSprinklerDgm();
ugm = convertToUgm(dgm);
model = ugm;
false = 1; true = 2;
C = 1; S = 2; R = 3; W = 4;
mW = marginal(model, W);
assert(approxeq(mW.T(true), 0.6471))

mSW = marginal(model, [S, W]);
assert(approxeq(mSW.T(true,true), 0.2781))

pSgivenW = pmf(marginal(model, S, W, true));
assert(approxeq(pSgivenW(true), 0.4298));
pSgivenWR = pmf(marginal(model, S, [W R], [true, true]));
assert(approxeq(pSgivenWR(true), 0.1945)); % explaining away

