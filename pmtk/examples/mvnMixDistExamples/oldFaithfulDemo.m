%% Fit a Mixture of Gaussians to the Old Faithful Data Set
cls;
setSeed(1);
load oldFaith;
m = fit(MvnMixDist('nmixtures',2),'data',X,'nrestarts',1);
pred = predict(m,X);
