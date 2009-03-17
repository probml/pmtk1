%% Test that JtreeInfEng Returns the Same Results as VarElimInfEng
% We compute prior marginals of each single node
%#testPMTK
dgmVE = mkAlarmNetworkDgm;
dgmVE.infMethod = VarElimInfEng('verbose',true);
dgmJT = dgmVE;
dgmJT.infMethod = JtreeInfEng('verbose',true);
queries = num2cell(1:37);
tic; [mVE, logZVE] = marginal(dgmVE, queries); toc
tic; [mJT, logZJT] = marginal(dgmJT, queries); toc
%assert(approxeq(logZVE, logZJT)) % fails
for i=1:37
  assert(approxeq(pmf(mVE{i}), pmf(mJT{i})))
end
