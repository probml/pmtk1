%% Test VarElimInfEng on the Alarm Network
%#slow

dgmVE = mkAlarmNetworkDgm();
dgmVE.infMethod = VarElimInfEng('verbose', false);
dgmJT = dgmVE;
dgmJT.infMethod = JtreeInfEng('verbose',false);
load alarmNetworkTest; % loads previously calculated marginal values to compare against. 
for i=1:37
    mVE = marginal(dgmVE, i, mod(i,37)+1, 2);
    mJT = marginal(dgmJT, i, mod(i,37)+1, 2);
    assert(approxeq(pmf(mVE),pmf(mJT),msaved{i}));
end

%% Large Marginal Test
dgm = mkAlarmNetworkDgm();
dgm.infMethod = VarElimInfEng();

%m = marginal(dgm,1:20); % 36 million entries!
m = marginal(dgm,1:5); 

