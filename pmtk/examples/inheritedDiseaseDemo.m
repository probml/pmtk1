%% Inherited Disease DGM Example 
%#testPMTK
G1 = 1; G2 = 2; G3 = 3;
X1 = 4; X2 = 5; X3 = 6;

%    X1
%    |
%    G1
%   /  \
%  G2   G3
%  |     |
%  X2    X3

% The Gi are binary (healthy or unhealthy gene),
% The Xi are cts observations
% If we observe X2 or X2,X3 the goal is to infer G1

graph = zeros(6);
graph(G1,X1) = 1;
graph(G1,G2) = 1;
graph(G1,G3) = 1;
graph(G2,X2) = 1;
graph(G3,X3) = 1;
%Graphlayout('adjMatrix',graph,'nodeLabels',{'G1','G2','G3','X1','X2','X3'})

% Prior on G1
CPD{G1} = TabularCPD([0.5;0.5]);

% Conditional G2|G1 and G3|G1
CPD{G2} = TabularCPD([0.9,0.1;0.1,0.9]);
CPD{G3} = CPD{G2};

% Observation model
XgivenG_H = MvnDist(50,10); % healthy
XgivenG_U = MvnDist(60,10); % unhealthy
%CPD{X1}   = MvnMixDist('distributions',{XgivenG_H,XgivenG_U});
CPD{X1}   = CondGaussCPD('-distributions',{XgivenG_H,XgivenG_U});
CPD{X2}   = CPD{X1};
CPD{X3}   = CPD{X1};

%dgm = DgmDist(graph,'CPDs', CPD,'infMethod', VarElimInfEng());
dgm = DgmDist(graph,'CPDs', CPD,'infMethod', JtreeInfEng());

evidence = {[50,50], [50], [60,60], [50,60]};
for i=1:length(evidence)
    ev = evidence{i};
    if length(ev)==2
        cond = [X2, X3];
    else
        cond = [X2];
    end
    [pG1a,logZ(i)] = marginal(dgm, G1, cond, ev); % a tabularFactor
    pG1b = pmf(pG1a); % a vector of numbers
    pG1(i) = pG1b(1); % probability in state 1 (healthy)
    %pG1(i) = sub(pmf(marginal(dgm, G1, cond, ev)),1); % 1 line version of above
end
assert(approxeq(pG1, [0.9863 0.8946 0.0137 0.5]))

