%% HMM vs DGM Comparing Forwards Backwards to Variable Elimination
% In this example, we compare a simple HMM model with binary hidden nodes and
% Gaussian emission densities to a DGM model equivalent up to the first T
% time steps. We show that running variable elimination on the DGM, to answer
% conditional queries about the hidden nodes, gives the same results as forwards
% backwards on the HMM.
%%
% The convention in this demo is that the latent variables will be labeled 1:T
% and the emission variables T+1:2*T.

T = 4;                          % Compare up to the first T timesteps
%% Setup the CPDs
transmat = [0.2,0.8;0.9,0.1];   % Transition matrix for the HMM, also the CPTs for the latent nodes
pi       = [0.5,0.5];           % Starting distribution for the HMM and DGM CPT for the first latent node
mvnOFF   = MvnDist(0  ,30);     % p(Yk | Zk=0) for all k
mvnON    = MvnDist(40 ,10);     % p(Yk | Zk=1) for all k
%% Create the HMM
hmm = HmmDist( 'startDist'     , DiscreteDist(pi')       ,...
               'transitionDist', DiscreteDist(transmat') ,...
               'emissionDist'  , {mvnOFF,mvnON}          );
%% Create the DGM
% We can call the convertToDgm method of the HMM class to create the DgmDist.
dgm = convertToDgm(hmm,T);                
%% Create the Evidence
setSeed(0);
evidence = squeeze(sample(hmm,10,T))';
evidence = [evidence;[25,25,25,25]];   
queries  = {1,2,3,4};

%% Condition on Evidence and Compare Marginals

for e=1:size(evidence,1)
    for q = 1:numel(queries)
        dgm       = condition(dgm ,T+1:2*T ,evidence(e,:));
        hmm       = condition(hmm ,'Y'     ,evidence(e,:));
        pQueryDGM = pmf(marginal (dgm, queries{q}));
        pQueryHMM = pmf(marginal( hmm, queries{q}));
        assert(approxeq(pQueryDGM,pQueryHMM));
    end
end
display(pQueryDGM);
display(pQueryHMM);

