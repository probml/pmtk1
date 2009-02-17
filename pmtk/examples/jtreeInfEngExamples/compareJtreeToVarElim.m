%% Test that JtreeInfEng Returns the Same Results as VarElimInfEng

dgmVE = mkAlarmNetworkDgm;
%dgmEnum = dgmVE;
%dgmEnum.infEng = EnumInfEng();
dgmJT = dgmVE;
dgmJT.infEng = JtreeInfEng();

[pVE,dgmVE] = marginal(dgmVE,32);
[pJT,dgmJT] = marginal(dgmJT,32);
%[pEnum,dgmEnum] = marginal(dgmEnum,[1,2]);
display(pmf(pVE));
display(pmf(pJT));
%display(pmf(pEnum));
