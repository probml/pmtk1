%% Test VarElimInfEng on the Alarm Network
%#slow

% This code does not check the answers are correct,
% merely that the functions can be called...
dgm = mkAlarmNetworkDgm();
%dgm.infMethod = 'varElim';
dgm.infMethod = VarElimInfEng('verbose', false);

for i=1:37
    m2 = marginal(dgm, i, mod(i,37)+2, 2);
end

%% Large Marginal Test
dgm = mkAlarmNetworkDgm();
dgm.infMethod = VarElimInfEng();

m = marginal(dgm,1:20); % 36 million entries!
%m = marginal(dgm,1:5); 

