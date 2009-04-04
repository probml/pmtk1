%% Simple Test of MvnMixDist
setSeed(13);
d = 2; K = 4;
%m = mkRndParams(MvnMixDist(),d,K);
m = mkRndParams(MixMvnEm('nmixtures', K, 'ndims', d));
X = sample(m,1000);
hold on;
plot(X(:,1),X(:,2),'.','MarkerSize',10);
%m1 = fit(MvnMixDist('nmixtures',4),'data',X,'nrestarts',3);
m = fit(m,X);
