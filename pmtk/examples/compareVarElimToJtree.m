%% Test that JtreeInfEng Returns the Same Results as VarElimInfEng
%#testPMTK
dgmVE = mkAlarmNetworkDgm;
dgmVE.infMethod = VarElimInfEng('verbose',false);
dgmJT = dgmVE;
dgmJT.infMethod = JtreeInfEng('verbose',false);
queries = num2cell(1:37);
mVE = marginal(dgmVE, queries);
mJT = marginal(dgmJT, queries);
for i=1:37
  assert(approxeq(pmf(mVE{i}), pmf(mJT{i})))
end
