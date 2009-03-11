%% Example of inference in water sprinkler DGM
%#testPMTK
dgm = mkSprinklerDgm();
false = 1; true = 2;
C = 1; S = 2; R = 3; W = 4;

% unconditional marginals
mW = pmf(marginal(dgm, W));
mSW = pmf(marginal(dgm, [S W]));
assert(approxeq(mW(true), 0.6471))
assert(approxeq(mSW(true,true), 0.2781))

% conditional marginals
pSgivenW = pmf(marginal(dgm, S, W, true));
assert(approxeq(pSgivenW(true), 0.4298));
pSgivenWR = pmf(marginal(dgm, S, [W R], [true, true]));
assert(approxeq(pSgivenWR(true), 0.1945)); % explaining away

% Display joint
joint = convertToJointTabularFactor(dgm);
joint = joint.T;
lab=cellfun(@(x) {sprintf('%d ',x)}, num2cell(ind2subv([2 2 2 2],1:16),2));
figure;
%bar(joint.T(:))
bar(joint(:))
set(gca,'xtick',1:16);
xticklabelRot(lab, 90, 10, 0.01)
title('joint distribution of water sprinkler DGM')
