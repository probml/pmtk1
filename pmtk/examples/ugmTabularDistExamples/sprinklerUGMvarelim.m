%% Variable Elimination in an Undirected Model
% In this example we test that the EnumInfEng and VarElimInfEng engines return
% the same results for both the dgm and ugm versions of the sprinkler network. 
%#testPMTK

dgmEnum         = mkSprinklerDgm();
dgmVE           = dgmEnum;
dgmEnum.infEng  = VarElimInfEng();
dgmVE.infEng    = VarElimInfEng();
ugmEnum         = convertToUgm(dgmEnum);
ugmVE           = ugmEnum;
ugmEnum.infEng  = EnumInfEng();
ugmVE.infEng    = VarElimInfEng();


false = 1; true = 2;
C = 1; 
S = 2; R = 3; W = 4;

models = {dgmVE,ugmEnum,ugmVE};

pW        = pmf(marginal(dgmEnum,W));
pSgivenW  = pmf(marginal(condition(dgmEnum,W,true),S));
pSgivenWR = pmf(marginal(condition(dgmEnum,[W,R],[true,true]),S));

for i=1:numel(models)
    pWtest        = pmf(marginal(models{i},W));
    pSgivenWtest  = pmf(marginal(condition(models{i},W,true),S));
    pSgivenWRtest = pmf(marginal(condition(models{i},[W,R],[true,true]),S));
    assert(approxeq(pWtest,pW));
    assert(approxeq(pSgivenWtest,pSgivenW));
    assert(approxeq(pSgivenWRtest,pSgivenWR));
end

