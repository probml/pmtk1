%% Test that JtreeInfEng Returns the Same Results as VarElimInfEng
%#testPMTK
dgmVE = mkAlarmNetworkDgm;
dgmJT = dgmVE;
dgmJT.infEng = JtreeInfEng();

for i=1:37
    [piVE,dgmVE] = marginal(dgmVE,i);
    [piJT,dgmJT] = marginal(dgmJT,i);
    assert(approxeq(pmf(piVE),pmf(piJT)));
end



