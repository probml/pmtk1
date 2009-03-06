%% Validate VarElimInfEng against EnumInfEng results
%#testPMTK
C = 1; S = 2; R = 3; W = 4;
% We compute every possible marginal of the sprinkler network
powerset = {[],C,S,R,W,[C,S],[C,R],[C,W],[S,R],[S,W],[R,W],[C,S,R],[C,S,W],[C,R,W],[S,R,W],[C,S,R,W]};
%% Marginal Test
for i=1:numel(powerset)
    dgmVE   = mkSprinklerDgm;
    dgmENUM = dgmVE;
    dgmVE.infEng = VarElimInfEng();
    dgmENUM.infEng = EnumInfEng();
    margVE = marginal(dgmVE,powerset{i});
    margENUM = marginal(dgmENUM,powerset{i});
    assert(approxeq(pmf(margVE),pmf(margENUM)));
end
%% Conditional Test 1
dgmVE = condition(dgmVE,[R,W],[1,1]);
pSgivenRW = marginal(dgmVE,S);
dgmENUM = condition(dgmENUM,[R,W],[1,1]);
pSgivenRW2 = marginal(dgmENUM,S);
assert(approxeq(pSgivenRW.T,pSgivenRW2.T));
%% Conditional Test 2
[dgm] = mkSprinklerDgm;
dgm.infEng = VarElimInfEng();
Tfac = convertToJointTabularFactor(dgm);
J = Tfac.T; % CSRW
C = 1; S = 2; R = 3; W = 4;
dgm = condition(dgm, [C W], [1 1]);
pSgivenCW = marginal(dgm, S);
pSgivenCW2 = sumv(J(1,:,:,1),3) ./ sumv(J(1,:,:,1),[2 3]);
assert(approxeq(colvec(pmf(pSgivenCW)), colvec(pSgivenCW2)))
X = sample(dgm, 100);
%% Lognormconst Test
lognormconst(dgm);
