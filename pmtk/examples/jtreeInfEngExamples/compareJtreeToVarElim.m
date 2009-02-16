%% Test that JtreeInfEng Returns the Same Results as VarElimInfEng

dgmVE = mkAlarmNetworkDgm;
dgmJT = dgmVE;
dgmJT.infEng = JtreeInfEng();

[p1VE,dgmVE] = marginal(dgmVE,4);
[p1JT,dgmJT] = marginal(dgmJT,4);
display(pmf(p1VE));
display(pmf(p1JT));