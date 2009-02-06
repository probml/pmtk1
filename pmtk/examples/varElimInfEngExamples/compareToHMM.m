%% HMM vs DGM Comparing Forwards Backwards to Variable Elimination
% In this example, we compare a simple HMM model with binary hidden nodes and
% Gaussian emission densities to a DGM model equivalent up to the first four
% time steps. We show that running variable elimination on the DGM, to answer
% conditional queries about the hidden nodes, gives the same results as forwards
% backwards on the HMM.
%% Label The Nodes
Z1 = 1; Z2 = 2;
Z3 = 3; Z4 = 4;
Y1 = 5; Y2 = 6;
Y3 = 7; Y4 = 8;

%% Create the Adjacency Matrix
graph = zeros(8);
graph(Z1,Z2) = 1; 
graph(Z2,Z3) = 1;
graph(Z3,Z4) = 1;
graph(Z1,Y1) = 1;
graph(Z2,Y2) = 1;
graph(Z3,Y3) = 1;
graph(Z4,Y4) = 1;
nodeLabels = {'Z1','Z2','Z3','Z4','Y1','Y2','Y3','Y4'};
%% Display the Graph
% Graphlayout('adjMatrix',graph,'nodeLabels',nodeLabels,'currentLayout',Treelayout());
%% Setup the CPDs
transmat = [0.2,0.8;0.9,0.1];   % Transition matrix for the HMM and DGM CPTs for Z2,Z3,Z4
pi       = [0.5,0.5];           % Starting distribution for the HMM and DGM CPT for Z1
mvnOFF   = MvnDist(0  ,30);     % p(Yk | Zk=0) for all k
mvnON    = MvnDist(40 ,10);     % p(Yk | Zk=1) for all k
obsCPD   = MvnMixDist('distributions',{mvnOFF,mvnON}); 

CPD{Z1} = TabularCPD(pi);
CPD{Z2} = TabularCPD(transmat);
CPD{Z3} = TabularCPD(transmat);
CPD{Z4} = TabularCPD(transmat);
CPD{Y1} = obsCPD;
CPD{Y2} = obsCPD;
CPD{Y3} = obsCPD;
CPD{Y4} = obsCPD;
%% Create the DGM
dgm      = DgmDist(graph,'CPDs',CPD,'domain',1:8,'infEng',VarElimInfEng());
%% Create the HMM
hmm      = HmmDist( 'startDist'     , DiscreteDist(pi')       ,...
                    'transitionDist', DiscreteDist(transmat') ,...
                    'emissionDist'  , {mvnOFF,mvnON}          );
%% Create the Evidence
%
setSeed(0);
evidence = squeeze(sample(hmm,10,4))';
evidence = [evidence;[25,25,25,25]];   
queries  = {Z1,Z2,Z3,Z4};

%% Condition on Evidence and Compare Marginals

for e=1:size(evidence,1)
    for q = 1:numel(queries)
        dgm       = condition(dgm ,[Y1,Y2,Y3,Y4],evidence(e,:));
        hmm       = condition(hmm ,'Y'          ,evidence(e,:));
        pQueryDGM = pmf(marginal (dgm, queries{q}));
        pQueryHMM =     marginal( hmm, queries{q});
        assert(approxeq(pQueryDGM,pQueryHMM));
    end
end
display(pQueryDGM);
display(pQueryHMM);

