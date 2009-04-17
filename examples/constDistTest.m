%% Simple Test of the ConstDist Class
%#testPMTK
point = 10*rand(20,1);
p = ConstDist(point);
logprob = logprob(p,point);
s = sample(p,10);
logZ = lognormconst(p);
nd = ndimensions(p);
m = mean(p);
v = var(p);
mm = mode(p);
plot(p);