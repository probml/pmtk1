%% Test that JtreeInfEng Returns the Same Results as VarElimInfEng

dgmVE = mkSprinklerDgm;
dgmJT = dgmVE;
dgmJT.infEng = JtreeInfEng();

[p1VE,dgmVE] = marginal(dgmVE,1);
[p1JT,dgmJT] = marginal(dgmJT,1);
display(pmf(p1VE));
display(pmf(p1JT));