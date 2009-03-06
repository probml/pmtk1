%% Test VarElimInfEng on the Alarm Network
%#slow

%% Alarm Network Test
dgm = mkAlarmNetworkDgm();
dgm.infEng = VarElimInfEng();
profile on;
for i=1:37
    m = marginal(dgm,[i,mod(i,37)+1]);
    dgm2 = condition(dgm,mod(i,37)+2,2);
    m2 = marginal(dgm2,i);
end
profile viewer;
%% Large Marginal Test
dgm = mkAlarmNetworkDgm();
dgm.infEng = VarElimInfEng();
profile on
m = marginal(dgm,1:20); % 36 million entries!
profile viewer
