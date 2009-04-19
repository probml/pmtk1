%% Simple Test of MixMvn
setSeed(13);
d = 2; K = 4;
m = MixMvn('nmixtures', K, 'ndims', d);
m = mkRndParams(m);
m.fitEng.verbose = true;
m.fitEng.nrestarts = 2;
m.fitEng.maxIter = 10;
X = sample(m,1000);
hold on;
plot(X(:,1),X(:,2),'.','MarkerSize',10);
%m1 = fit(MvnMixDist('nmixtures',4),'data',X,'nrestarts',3);
m = fit(m,X);
