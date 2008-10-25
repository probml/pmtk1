% QMRstub

Dnodes = [1 2 3];
Snodes = [4 5 6 7 8];
N = 8;
G = zeros(3, 5);
G(1,[1 2 4]) = 1;
G(2,2) = 1;
G(3,[3 4 5]) = 1;

dprior = [0.1 0.1 0.1];
inhibit = zeros(3,5);
leakInhibit = ones(1,5);
%leakInhibit = 0.999*ones(1,5);

bnet = mk_qmr_bnet(G, inhibit, leakInhibit, dprior);
engine = jtree_inf_engine(bnet);

neg = 1; pos = 2;
evidence = cell(1,8);
evidence(Snodes(1)) = pos; % first example
[engine, ll] = enter_evidence(engine, evidence);
%%% fill in the rest of the code
%%% to extract posterior marginals on the disease nodes
%%% for the 4 cases
