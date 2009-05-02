%% Fit Mixture of Gaussians with different number of mixture components
%#broken
% Compare to kmeansModelSelDemo2d

setSeed(0);
d = 10; K = 4;
Ntrain = 1000; Ntest = 1000;

M = MixMvn(K, d);
M = mkRndParams(M);
Xtrain = sample(M, Ntrain);
Xtest = sample(M, Ntest);

Ks = [2 4 6 8];
Nm = length(Ks);
Ms = cell(1,Nm);
for m=1:Nm
  Ms{m} = MixMvn(Ks(m), d);
end
ML = ModelList(Ms, 'BIC');
ML = fit(ML, Xtrain);

figure; plot(Ks, ML.loglik, 'o-');title('train LL')
figure; plot(Ks, ML.penloglik, 'o-'); title('train BIC')

% Sanity check
ll = sum(logprob(ML, Xtrain), 1);
Kbest = nmixtures(ML.bestModel);
KbestNdx = find(Kbest==Ks);
assert(isequal(ll, ML.loglik(KbestNdx)))

% Compute test set LL
for m=1:length(ML.models)
  ll(m) = sum(logprob(ML.models{m}, Xtest),1);
end
figure; plot(Ks, ll, 'o-'); title('test LL')



  
  

  