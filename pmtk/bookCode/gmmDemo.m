% from
%
% http://www.mathworks.com/access/helpdesk/help/toolbox/stats/index.html?/access/helpdesk/help/toolbox/stats/bq_679x-24.html&http://www.mathworks.com/cgi-bin/texis/webinator/search/?db=MSS&prox=page&rorder=750&rprox=750&rdfreq=500&rwfreq=500&rlead=250&sufs=0&order=r&is_summary_on=1&ResultCount=10&query=gaussian+mixture

seed = 0; rand('state', seed); randn('state', seed);
mu1 = [1 2];
sigma1 = [2 0; 0 .5];
mu2 = [-3 -5];
sigma2 = [1 0; 0 1];
X = [mvnrnd(mu1,sigma1,1000);mvnrnd(mu2,sigma2,1000)];

figure(1);clf
scatter(X(:,1),X(:,2),10,'.')
hold on

options = statset('Display','final');
obj = gmdistribution.fit(X,2,'Options',options);

h = ezcontour(@(x,y)pdf(obj,[x y]),[-8 6],[-8 6]);


obj = cell(1,4);
for k = 1:4
    obj{k} = gmdistribution.fit(X,k);
    AIC(k)= obj{k}.AIC;
    BIC(k)= obj{k}.BIC;
end

 
