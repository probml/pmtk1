%% Simple Test of the BernoulliDist Class
%#testPMTK
m = BernoulliDist();
X = rand(10,2)>0.5;
m = fit(m, X);