%% plot some IW distributions
seed=1;randn('state',seed);rand('twister',seed);
S=randpd(2);
R=cov2cor(S)
p=InvWishartDist(20,S);
mean(p)
figure(1);clf;plotSamples2d(p,9);
figure(2);clf;plotMarginals(p)