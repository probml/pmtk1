%% Inference in undirected version of sprinkler network
% Compare to sprinklerDGMdemo


dgm = mkSprinklerDgm();
ugm = convertToUgm(dgm);
ugm.infEng  = EnumInfEng();
model = ugm;
false = 1; true = 2;
C = 1; S = 2; R = 3; W = 4;
mW = marginal(model, W);
assert(approxeq(mW.T(true), 0.6471))

% batch queries
marg = marginal(model, {W, [S W]});
mW = marg{1}; mSW = marg{2};
assert(approxeq(mW.T(true), 0.6471))
assert(approxeq(mSW.T(true,true), 0.2781))

pSgivenW = predict(model, W, true, S);
assert(approxeq(pSgivenW.T(true), 0.4298));
pSgivenWR = predict(model, [W R], [true, true], S);
assert(approxeq(pSgivenWR.T(true), 0.1945)); % explaining away

