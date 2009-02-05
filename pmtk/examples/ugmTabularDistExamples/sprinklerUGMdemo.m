%% Inference in undirected version of sprinkler network
% Compare to sprinklerDGMdemo
%#testPMTK

dgm = mkSprinklerDgm();
ugm = convertToUgm(dgm);
ugm.infEng  = EnumInfEng();
model = ugm;
false = 1; true = 2;
C = 1; S = 2; R = 3; W = 4;
mW = marginal(model, W);
assert(approxeq(mW.T(true), 0.6471))

mW = marginal(model, [S, W]);
assert(approxeq(mW.T(true,true), 0.2781))

model = condition(model, W, true);
%pSgivenW = predict(model, W, true, S);
pSgivenW = marginal(model, S);
assert(approxeq(pSgivenW.T(true), 0.4298));
%pSgivenWR = predict(model, [W R], [true, true], S);
model = condition(model, [W R], [true, true]);
pSgivenWR = marginal(model, S);
assert(approxeq(pSgivenWR.T(true), 0.1945)); % explaining away

