%% Perform Gibbs sampling for a Mixture of Gaussians on the Old Faithful Data Set
%author Cody Severinski
setSeed(1);
load oldFaith;
[n d] = size(X);
%meantrue = mean(X);

% Gibbs
m = MixMvnGibbs('nmixtures',2);
dists = latentGibbsSample(m, X, 'verbose', true);
%meansample = mean(dists.muDist);
predictGibbs = predict(dists.latentDist);
postGibbs = mode(predictGibbs);

% EM
m  = MixMvn('-nmixtures',2,'-ndims',d);
m = fit(m,'-data',X);
predictEM = inferLatent(m,X);
postEM = mode(predictEM);

% We need to do this because EM and Gibbs numbers the clusters differently.
% This can (and perhaps should) be made into a function somewhere.
K = max(dists.latentDist.support);
agree = zeros(K,K);
for i=1:K
  for j=1:K
    gibbsidx = find(postGibbs == i);
    emidx = find(postEM == j);
    agree(i,j) = length(intersectPMTK(gibbsidx, emidx));
  end
end

[row,col] = find(bsxfun(@eq,agree,max(agree, [], 2)));
predictGibbs.T = predictGibbs.T(row,:);

absdiff = abs(predictEM.T(1,:) - predictGibbs.T(1,:));

plot(absdiff, 'o', 'linewidth', 3);
title( sprintf('Difference in p(x_i = 1).  Mean = %g.  Median = %g', mean(absdiff), median(absdiff)) );
