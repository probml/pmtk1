%% Fit a Mixture of Gaussians to the Old Faithful Data Set
%#testPMTK
setSeed(1);
load oldFaith;
[n d] = size(X);
%m = fit(MvnMixDist('nmixtures',2),'data',X,'nrestarts',1);
m  = MixMvn('nmixtures',2,'ndims',d);
m = fit(m,'data',X);
%pred = predict(m,X);
post = inferLatent(m,X);