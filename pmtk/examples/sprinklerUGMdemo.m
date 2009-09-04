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

for i=1:4
  CPD{i} = TabularFactor(dgm.CPDs{i}.T, family(dgm.G,i));
end
fac{1} = TabularFactor(ones(2,2,2), [C S R]);
for i=1:3
  fac{1} = multiplyBy(fac{1}, CPD{i});
end
fac{2} = CPD{4};
jointF = TabularFactor.multiplyFactors(fac)
T = jointF.T

joint = convertToJointTabularFactor(ugm);
joint = joint.T;
lab=cellfun(@(x) {sprintf('%d ',x)}, num2cell(ind2subv([2 2 2 2],1:16),2));
figure;
%bar(joint.T(:))
bar(joint(:))
set(gca,'xtick',1:16);
xticklabelRot(lab, 90, 10, 0.01)
title('joint distribution of water sprinkler UGM')


