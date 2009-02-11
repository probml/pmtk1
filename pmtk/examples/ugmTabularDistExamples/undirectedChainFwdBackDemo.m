%% Run Forwards Backwards on an Undirected Chain
% In this example, we run forwards backwards on an undirected chain and compare
% the results to running variable elimination instead. 
%#testPMTK

A = 1; B = 2; C = 3; D = 4; E = 5;
G = zeros(5);
G(A,B) = 1;
G(B,C) = 1;
G(C,D) = 1;
G(D,E) = 1;
G = mkSymmetric(G);
T = [14, 4 ;29 ,144];
pi = [100,20];
factors{1} = TabularFactor(pi ,A);
factors{2} = TabularFactor(T ,[A,B]);
factors{3} = TabularFactor(T ,[B,C]);
factors{4} = TabularFactor(T ,[C,D]);
factors{5} = TabularFactor(T ,[D,E]);
ugm =  UgmTabularDist('G', G, 'factors', factors,'infEng',VarElimInfEng());

pEvarelim = pmf(marginal(ugm,E));
evidence = ones(size(T,1),5);                                 % e.g. no evidence

[gamma,alpha,beta,logp] = hmmFwdBack(pi,T,evidence);
pEfwdBack = gamma(:,E);
assert(approxeq(pEvarelim,pEfwdBack));