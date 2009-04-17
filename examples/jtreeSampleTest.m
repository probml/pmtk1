%% JTreeInfEng Forwards Filtering, Backwards Sampling
% Here we compare the sampling implementation in JtreeInfEng to forwards
% samplings in DgmDist
%#testPMTK

setSeed(0);
dgm = mkSprinklerDgm();
%dgm = mkAlarmNetworkDgm;
dgm.infMethod = JtreeInfEng();
n = 1000;
%% 
% Samples using forwards sampling.
S1 = sample(dgm,n);
%%
% The DgmDist delegates calls to sample() involving evidence. We provide
% empty evidence here, simply to access the right function for testing
% purposes. 
S2 = sample(dgm,n,[],[]);
hist1 = normalize(sum(S1==1));
hist2 = normalize(sum(S2==1));
assert(approxeq(hist1,hist2,0.1));
figure; bar(hist1); 
title('Forwards Sampling');
figure; bar(hist2); 
title('Forwards Filtering, Backwards Sampling');
