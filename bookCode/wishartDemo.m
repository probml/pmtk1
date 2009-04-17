% based on the example at
%% http://www.mathworks.com/access/helpdesk/help/toolbox/stats/index.html?/access/helpdesk/help/toolbox/stats/f75080.html&http://www.mathworks.com/access/helpdesk/help/toolbox/stats/helptoc.html

seed = 0; rand('state', seed); randn('state', seed);
mu = [0 0];
Sigma = [1 .5; .5 2];

n = 1000;
X = mvnrnd(mu,Sigma,n);
cov(X)
S2 = wishrnd(Sigma,n)/(n-1)
