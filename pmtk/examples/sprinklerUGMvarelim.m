%% Variable Elimination in an Undirected Model
% In this example we test that the EnumInfEng and VarElimInfEng engines return
% the same results for both the dgm and ugm versions of the sprinkler network. 
%#testPMTK


dgm = mkSprinklerDgm();
dgmVE = dgm; dgmVE.infMethod = VarElimInfEng;
dgmEnum = dgm; dgmEnum.infMethod = EnumInfEng;
ugm = convertToUgm(dgm);
ugmVE = ugm; ugmVE.infMethod = VarElimInfEng;
ugmEnum = ugm; ugmEnum.infMethod = EnumInfEng;

false = 1; true = 2;
C = 1; 
S = 2; R = 3; W = 4;

models = {dgmVE, ugmEnum,ugmVE};

pW        = pmf(marginal(dgmEnum,W));
pSgivenW  = pmf(marginal(dgmEnum,S,W,true));
pSgivenWR = pmf(marginal(dgmEnum,S,[W,R],[true,true]));

for i=1:numel(models)
    pWtest        = pmf(marginal(models{i},W));
    pSgivenWtest  = pmf(marginal(models{i},S, W, true));
    pSgivenWRtest = pmf(marginal(models{i},S, [W,R], [true,true]));
    assert(approxeq(pWtest,pW));
    assert(approxeq(pSgivenWtest,pSgivenW));
    assert(approxeq(pSgivenWRtest,pSgivenWR));
end

