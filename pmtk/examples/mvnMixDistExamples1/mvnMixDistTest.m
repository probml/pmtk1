%% Simple Test of MvnMixDist
setSeed(13);
m = mkRndParams(MvnMixDist(),2,4);
X = sample(m,1000);
hold on;
plot(X(:,1),X(:,2),'.','MarkerSize',10);
m1 = fit(MvnMixDist('nmixtures',4),'data',X,'nrestarts',5);
