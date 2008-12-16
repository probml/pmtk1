%%  Posterior Predictive Distribution for Beta-Binomial model
N = 10;
X = [1 2]';
N1 = sum(X);
N0 = N*length(X)-N1;
a = 2;
b = 2;
m = Binom_BetaDist(N, BetaDist(a,b));
m0 = predict(m);
figure; plot(m0); title('prior predictive')
m = fit(m, 'data', X);
mm = predict(m); % posterior predictive is BetaBinomDist
figure;
h1 = plot(mm, 'plotArgs', 'b'); title('posterior predictive')
% MAP estimation
m3 = fit(BinomDist(N, []), 'data', X, 'prior', BetaDist(a,b));
%mm3 = predict(m3); % BinomDist
mm3 = m3; 
hold on
h3 = plot(mm3);
set(h3, 'faceColor','r','barwidth',0.5);
legend('postpred', 'plugin');