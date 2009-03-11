%% Test VarElimInfEng on the Alarm Network
%#slow

% This code does not check the answers are correct,
% merely that the functions can be called...
dgm = mkAlarmNetworkDgm();
%dgm.infMethod = 'varElim';
dgm.infEng = VarElimInfEng('verbose', false);
profile on;
for i=1:37
    m2 = marginal(dgm, i, mod(i,37)+2, 2);
end


profile viewer;
%% Large Marginal Test
dgm = mkAlarmNetworkDgm();
dgm.infEng = VarElimInfEng();
profile on
m = marginal(dgm,1:20); % 36 million entries!
%m = marginal(dgm,1:5); 
profile viewer
