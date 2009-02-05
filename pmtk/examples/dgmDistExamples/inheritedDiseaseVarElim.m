%% Inherited Disease DGM Example
%#testPMTK
G1 = 1; G2 = 2; G3 = 3;
X1 = 4; X2 = 5; X3 = 6;

graph = zeros(6);
graph(G1,X1) = 1;
graph(G1,G2) = 1;
graph(G1,G3) = 1;
graph(G2,X2) = 1;
graph(G3,X3) = 1;
%Graphlayout('adjMatrix',graph,'nodeLabels',{'G1','G2','G3','X1','X2','X3'},'currentLayout',TreeLayout())

% Prior on G1
CPD{G1} = TabularCPD([0.5;0.5]); 

% Conditional G2|G1 and G3|G1
CPD{G2} = TabularCPD([0.9,0.1;0.1,0.9]);
CPD{G3} = CPD{G2};

% Observation model
XgivenG1 = MvnDist(50,10); % healthy
XgivenG2 = MvnDist(60,10); % unhealthy
CPD{X1}  = MvnMixDist('distributions',{XgivenG1,XgivenG2});
CPD{X2}  = CPD{X1};
CPD{X3} = CPD{X1};

dgm = DgmDist(graph,'CPDs', CPD,'infEng',VarElimInfEng(),'domain',1:6);

evidence = {[50,50], [50], [60,60], [50,60]};
for i=1:length(evidence)
  ev = evidence{i};
  if length(ev)==2
    dgm  = condition(dgm,[X2,X3],ev);
  else
    dgm  = condition(dgm,[X2],ev);
  end
  pG1VE(i) = sub(pmf(marginal(dgm,G1)),1); 
end
