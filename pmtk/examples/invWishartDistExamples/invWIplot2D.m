%% plot some IW distributions
setSeed(1);
S=randpd(2);
R=cov2cor(S);
p=InvWishartDist(20,S);
mean(p)
plotSamples2d(p,9);
plotMarginals(p)
restoreSeed();