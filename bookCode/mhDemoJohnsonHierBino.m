function mhDemoJohnsonHierBino()
% Hierarchical Bayesian estimation of some binomial proportions
% Johnson and Albert  p67

% data p24
data.y = [0 0 2 0 1 1 ...
	  0 2 1 3 0 1 ...
	  1 1 54 0 0 1 ...
	  3 0];

data.n = [1083 855 3461 657 1208 1025 ...
	  527 1668 583 582 917 857 ...
	  680 917 53637 874 395 581 ...
	  588 383];

% EB matrix of counts
X = [data.y(:) data.n(:)-data.y(:)];
% = [data.n(:)-data.y(:) data.y(:)];
alphas = polya_fit(X) 
alphas = polya_fit_simple(X)
%lphas = polya_fit_ms(X)
alphas = polya_moment_match(X)


hparams.am = 0.01;
hparams.bm = 9.99; % JA p25 says 9.9, but then prior precision is not 10

Nsamples = 5000;       
x = zeros(Nsamples,2); % x(s,:)  = [theta1, theta2]
sigma_prop = 0.3*eye(2); % if smaller, accept rate gets too high     

targetArgs = {data, hparams};
proposalArgs = {sigma_prop};

seed = 1; randn('state', seed); rand('state', seed);
xinit = 0.1*randn(1,2); % initial state
[x, naccept] = metrop(@target, @proposal, xinit, Nsamples,  targetArgs, proposalArgs);

figure(1);clf
plot(x(:,1), 'r-');
hold on
plot(x(:,2), 'b:');
title(sprintf('red=logitM, blue=logK, accept rate = %5.3f', naccept/Nsamples));

burnin = 500;
samples.m = sigmoid(x(burnin:end,1));
samples.K = exp(x(burnin:end,2));

Kmean = mean(samples.K);
mMean = mean(samples.m);
aMean = Kmean*mMean;
bMean = Kmean*(1-mMean);
[aMean bMean]
for i=1:length(samples.m)
  aPost(i) = samples.K(i)*samples.m(i);
  bPost(i) = samples.K(i)*(1-samples.m(i));
end
[mean(aPost) mean(bPost)]


d = length(data.n); % ncities;
samples.theta = zeros(Nsamples-burnin+1, d);
for i=1:d
  as = data.y(i) + samples.K .* samples.m;
  bs = data.n(i) - data.y(i) + samples.K .* (1-samples.m);
  samples.theta(:,i) = betarnd(as, bs);
end
for i=1:d
  post.meantheta(i) = mean(samples.theta(:,i));
  thetaMLE(i) = data.y(i)/data.n(i);
end
thetaPooledMLE = sum(data.y)/sum(data.n);


figure(2);clf
subplot(4,1,1); bar(data.y); title('number of people with cancer (truncated at 5)')
set(gca,'ylim',[0 5])
subplot(4,1,2); bar(data.n); title('pop of city (truncated at 2000)');
set(gca,'ylim',[0 2000])
subplot(4,1,3); bar(thetaMLE);title('MLE');
subplot(4,1,4); bar(post.meantheta);title('posterior mean (red line=pooled MLE)')
hold on;h=line([0 20], [thetaPooledMLE thetaPooledMLE]);
set(h,'color','r','linewidth',2)

keyboard

%%%%%%%%%%

function logp = target(x, data, hparams)

logitM = x(1); logK = x(2);
m = sigmoid(logitM); K = exp(logK);
ncases = length(data.n);
logp = (hparams.am-1)*log(m) + (hparams.bm-1)*log(1-m) ...
       -2*log(1+K) + sum(logbeta(K*m+data.y, K*(1-m)+data.n-data.y)) ...
       -ncases*logbeta(K*m, K*(1-m));


function logp = logbeta(a,b)
logp = gammaln(a) + gammaln(b) - gammaln(a+b);


function xp = proposal(x, Sigma)
d = size(Sigma,1);
xp = x + mvnrnd(zeros(1,d), Sigma);


