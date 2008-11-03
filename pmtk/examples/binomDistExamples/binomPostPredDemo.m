%% Bayesian Inference - Posterior Predictive Distribution
N = 10;
X = [1 2];
N1 = sum(X);
N0 = N*length(X)-N1;
[N1 N0]
a = 2;
b = 2;
m = binomDist(N, betaDist(a,b));
%m = bernoulliDist(betaDist(a,b));
prior = m.mu; % betaDist
m0 = postPredict(m); figure; plot(m0); title('prior predictive')
m = inferParams(m, 'data', X);
mm = postPredict(m); % posterior predictive is betaBinomDist
figure;
h1 = plot(mm, 'plotArgs', 'b'); title('posterior predictive')
% MAP estimation
m3 = binomDist(N, betaDist(a,b));
m3 = fit(m3, 'data', X, 'method', 'map');
%m3.mu % constant vector
mm3 = predict(m3); % binomDist
hold on
h3 = plot(mm3);
set(h3, 'faceColor','r','barwidth',0.5);
legend('postpred', 'plugin');