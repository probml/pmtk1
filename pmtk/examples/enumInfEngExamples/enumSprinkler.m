%% Test EnumInfEng 
%#testPMTK
[dgm] = mkSprinklerDgm;
dgm.infEng = EnumInfEng();
Tfac = convertToJointTabularFactor(dgm);
J = Tfac.T; % CSRW
C = 1; S = 2; R = 3; W = 4;
dgm = condition(dgm, [C W], [1 1]);
pSgivenCW = marginal(dgm, S);
pSgivenCW2 = sumv(J(1,:,:,1),3) ./ sumv(J(1,:,:,1),[2 3]);
assert(approxeq(pSgivenCW.T(:), pSgivenCW2(:)))
X = sample(dgm, 100);