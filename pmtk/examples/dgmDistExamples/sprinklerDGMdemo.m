%% Example of explaining away
%#broken
dgm = mkSprinklerDgm;
dgm  = initInfEng(dgm, 'enumeration');
false = 1; true = 2;
C = 1; S = 2; R = 3; W = 4;

mW = marginal(dgm, W);
assert(approxeq(mW.T(true), 0.6471))
mSW = marginal(dgm, [S W]);
assert(approxeq(mSW.T(true,true), 0.2781))

pSgivenW = predict(dgm, W, true, S);
assert(approxeq(pSgivenW.T(true), 0.4298));
pSgivenWR = predict(dgm, [W R], [true, true], S);
assert(approxeq(pSgivenWR.T(true), 0.1945)); % explaining away

% Display joint
joint = dgmDiscreteToTable(dgm);
lab=cellfun(@(x) {sprintf('%d ',x)}, num2cell(ind2subv([2 2 2 2],1:16),2));
figure;
%bar(joint.T(:))
bar(joint(:))
set(gca,'xtick',1:16);
xticklabelRot(lab, 90, 10, 0.01)
title('joint distribution of water sprinkler DGM')