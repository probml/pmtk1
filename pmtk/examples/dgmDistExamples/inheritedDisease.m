

G1 = 1; G2 = 2; G3 = 3;
X1 = 4; X2 = 5; X3 = 6;

graph = zeros(6);
graph(G1,X1) = 1;
graph(G1,G2) = 1;
graph(G1,G3) = 1;
graph(G2,X2) = 1;
graph(G3,X3) = 1;
%Graphlayout('adjMatrix',graph,'nodeLabels',{'G1','G2','G3','X1','X2','X3'},'currentLayout',TreeLayout())


CPD{G1} = TabularCPD([0.5;0.5]);
CPD{G2} = TabularCPD([0.9,0.1;0.1,0.9]);
CPD{G3} = CPD{G2};

XgivenG0 = MvnDist(50,10);
XgivenG1 = MvnDist(100,10);
CPD{X1}  = MvnMixDist('distributions',{XgivenG0,XgivenG1});
CPD{X2}  = CPD{X1};

XgivenG0 = MvnDist(75,10);
XgivenG1 = MvnDist(75,10);
CPD{X3}  = MvnMixDist('distributions',{XgivenG0,XgivenG1});

CPD{X3} = CPD{X1}

dgm = DgmDist(graph,'CPDs', CPD,'infEng',VarElimInfEng(),'domain',1:6);

dgm1  = condition(dgm,[X2,X3],[50,50]) ;

pG1marg = sub(pmf(marginal(dgm1,G1)),1) % p(G1 = 0 | X2 = 50, X3 = 50)


% dgm2 = condition(dgm,[X2,X3],[100,100]);
% pG1marg = marginal(dgm2,G1);
% p = pG1marg.T(1)     % p(G1 = 0 | X2 = 100, X3 = 100)

% 
% dgm3 = condition(dgm,[X2,X3],[50,100]);
% pG1marg = sub(pmf(marginal(dgm3,G1)),1)
