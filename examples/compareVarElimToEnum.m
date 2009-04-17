%% Validate VarElimInfEng against EnumInfEng results
%#testPMTK
C = 1; S = 2; R = 3; W = 4;
% We compute every possible marginal of the sprinkler network
powerset = {[],C,S,R,W,[C,S],[C,R],[C,W],[S,R],[S,W],[R,W],[C,S,R],[C,S,W],[C,R,W],[S,R,W],[C,S,R,W]};
%% Marginal Test
dgmVE   = mkSprinklerDgm;
dgmENUM = dgmVE;
dgmVE.infMethod = VarElimInfEng(); 
dgmENUM.infMethod = EnumInfEng();
for i=1:numel(powerset)
    margVE = marginal(dgmVE,powerset{i});
    margENUM = marginal(dgmENUM,powerset{i});
    assert(approxeq(pmf(margVE),pmf(margENUM)));
end
%% Conditional Test 1
pSgivenRW = marginal(dgmVE, S, [R,W], [1,1]);
pSgivenRW2 = marginal(dgmENUM, S, [R,W], [1,1]);
assert(approxeq(pmf(pSgivenRW),pmf(pSgivenRW2)));
assert(approxeq(pmf(pSgivenRW),[0.9325;0.0675]));
%% Conditional Test 2
[dgm] = mkSprinklerDgm;
Tfac = convertToJointTabularFactor(dgm);
J = Tfac.T; % CSRW
C = 1; S = 2; R = 3; W = 4;
pSgivenCW2 = sumv(J(1,:,:,1),3) ./ sumv(J(1,:,:,1),[2 3]);
dgms = {dgmVE, dgmENUM};
for i=1:length(dgms)
  dgm = dgms{i};
  pSgivenCW = marginal(dgm, S, [C W], [1 1]);
  assert(approxeq(colvec(pmf(pSgivenCW)), colvec(pSgivenCW2)))
end
