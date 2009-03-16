%% UGM vs DGM Chains of Different Length
% In this example, we examine how the length of a chain effects marginals in a
% UGM, (so long as the factors are not locally normalized), but not in a DGM. 
%% Setup
smallSize = 5;
bigSize   = 10;
queryNode = 2;   
nstates   = 4;

Gsmall = diag(ones(1,smallSize-1),1);
Gbig   = diag(ones(1,bigSize-1),1);

T  = normalize(rand(nstates),2);     % Tied transition probabilities
pi = normalize(rand(1,nstates));     % p(X1)

CPDsmall{1}           = TabularCPD(pi);
CPDsmall(2:smallSize) = copy(TabularCPD(T),1,smallSize-1); 

CPDbig{1}           = TabularCPD(pi);
CPDbig(2:bigSize)   = copy(TabularCPD(T),1,bigSize-1); 

DGMsmall = DgmDist(Gsmall,'CPDs',CPDsmall,'domain',1:smallSize);
DGMbig   = DgmDist(Gbig  ,'CPDs',CPDbig  ,'domain',1:bigSize);
%% DGM Comparison
% Compare the marginals PX2 in both chains. 
fprintf('\n Small vs Long DGMs\n\n');
pX2_DGM_Small = pmf(marginal(DGMsmall,queryNode))
pX2_DGM_Big   = pmf(marginal(DGMbig  ,queryNode))
diff = sum(abs(pX2_DGM_Small - pX2_DGM_Big))
pause(1);
%% Converted UGMs
% Here we simply convert the DGMs to UGMs and perform the same marginalization.
% Since the potentials are locally normalized, however, the results match those
% of the DGM.
UGMsmall = convertToUgm(DGMsmall);
UGMbig   = convertToUgm(DGMbig);

fprintf('\n Small vs Long Locally Normalized UGMs\n\n');
pX2_UGM_Small = pmf(marginal(UGMsmall,queryNode))
pX2_UGM_Big   = pmf(marginal(UGMbig  ,queryNode))
diff          = sum(abs(pX2_UGM_Small - pX2_UGM_Big))
pause(1);
%% Remove pi Factor
% We now show that removing the factor on X1, (i.e. the prior on X1) does not
% cause the marginals from the two chains to differ.  
UGMsmall.factors = UGMsmall.factors(2:end);
UGMbig.factors   = UGMbig.factors(2:end);

fprintf('\n Small vs Long Locally Normalized UGMs with no Factor on X1\n\n');
pX2_UGM_Small2 = pmf(marginal(UGMsmall,queryNode))
pX2_UGM_Big2   = pmf(marginal(UGMbig  ,queryNode))
diff           = sum(abs(pX2_UGM_Small2 - pX2_UGM_Big2))
pause(1);
%% Test UGMs with Unnormalized Factors
% Here we notice a small difference. 
T2  = 100*rand(nstates);
TfacsSmall = cell(1,smallSize-1);
for i=1:smallSize-1
    TfacsSmall{i} = TabularFactor(T2,[i,i+1]);
end

TfacsBig = cell(1,bigSize-1);
for i=1:bigSize-1
    TfacsBig{i} = TabularFactor(T2,[i,i+1]);
end

UGMsmall2 = UgmTabularDist('factors',TfacsSmall,'nstates',nstates*ones(1,smallSize));
UGMbig2   = UgmTabularDist('factors',TfacsBig  ,'nstates',nstates*ones(1,bigSize));
fprintf('\n Small vs Long UGMs\n\n');
pX2_UGM_Small3 = pmf(marginal(UGMsmall2,queryNode))
pX2_UGM_Big3   = pmf(marginal(UGMbig2,queryNode))

diff = sum(abs(pX2_UGM_Small3 - pX2_UGM_Big3))
