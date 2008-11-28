function mhMissouriCancer()
% Hierarchical Bayesian estimation of some binomial proportions
% Johnson and Albert  p67
% mhDemoJohnsonHierBino C:\kmurphy\pmtkLocal\BLTold\PMLcode\Book

% data p24
data.y = [0 0 2 0 1 1 ...
	  0 2 1 3 0 1 ...
	  1 1 54 0 0 1 ...
	  3 0];

data.n = [1083 855 3461 657 1208 1025 ...
	  527 1668 583 582 917 857 ...
	  680 917 53637 874 395 581 ...
	  588 383];

hparams.am = 0.01;
hparams.bm = 9.99; % JA p25 says 9.9, but then prior precision is not 10

Nsamples = 5000;       
SigmaProp = 0.3*eye(2); % if smaller, accept rate gets too high     
setSeed(1); 
xinit = 0.1*randn(1,2); % initial state
proposal = @(x) (x + mvnrnd(zeros(1,2), SigmaProp));

[x, naccept] = metropolisHastings(@target, proposal, xinit, Nsamples);

% trace plot
figure;
plot(x(:,1), 'r-');
hold on
plot(x(:,2), 'b:');
title(sprintf('red=logitM, blue=logK, accept rate = %5.3f', naccept/Nsamples));


burnin = 500;
samples.m = sigmoid(x(burnin:end,1));
samples.K = exp(x(burnin:end,2));
samples.logitm = x(burnin:end, 1);
samples.logK = x(burnin:end, 2);

% plot of posterior
figure;plot(samples.logitm, samples.logK, '.')
xlabel('logitm'); ylabel('log K')

%figure;plot(samples.m, samples.K, '.')
%xlabel('m'); ylabel('K')

[xs, ys] = meshgrid(-8:0.05:-5, 4:0.1:12);
xy = [xs(:)'; ys(:)']';
p = exp(target(xy));
p = reshape(p, size(xs));
h = max(p(:));
vals = [0.1*h 0.01*h 0.001*h];
figure; contour(xs(1,:), ys(:,1), p, vals);
figure; imagesc(p);

% posterior of theta(i) given m,K
d = length(data.n); % ncities;
samples.theta = zeros(Nsamples-burnin+1, d);
for i=1:d
  as = data.y(i) + samples.K .* samples.m;
  bs = data.n(i) - data.y(i) + samples.K .* (1-samples.m);
  samples.theta(:,i) = betarnd(as, bs);
  post.meantheta(i) = mean(samples.theta(:,i));
  thetaMLE(i) = data.y(i)/data.n(i);
end
thetaPooledMLE = sum(data.y)/sum(data.n);


figure;
subplot(4,1,1); bar(data.y); title('number of people with cancer (truncated at 5)')
set(gca,'ylim',[0 5])
subplot(4,1,2); bar(data.n); title('pop of city (truncated at 2000)');
set(gca,'ylim',[0 2000])
subplot(4,1,3); bar(thetaMLE);title('MLE');
subplot(4,1,4); bar(post.meantheta);title('posterior mean (red line=pooled MLE)')
hold on;h=line([0 20], [thetaPooledMLE thetaPooledMLE]);
set(h,'color','r','linewidth',2)

keyboard


  function logp = target(x)
    logitM = x(:,1); logK = x(:,2);
    m = sigmoid(logitM); K = exp(logK);
    ncases = length(data.n);
    logp = (hparams.am-1).*log(m) + (hparams.bm-1).*log(1-m) ...
      -2*log(1+K) + sum(betaln(K*m+data.y, K*(1-m)+data.n-data.y)) ...
      -ncases*betaln(K*m, K*(1-m));
  end


end
