function decisionBoundariesDemo1d

figure(1); clf

subplot(2,2,1)
plotgaussians1d(-1, 3, 1, 1, 0.5)
subplot(2,2,2)
plotgaussians1d(-1, 3, 1, 1, 0.1)
subplot(2,2,3)
plotgaussians1d(1, -1, 1, 1, 0.5)
subplot(2,2,4)
plotgaussians1d(1, -1, 3, 1, 0.5)

function plotgaussians1d(mu1, mu2, s1, s2, pi1)
pi2 = 1-pi1;
xs = -5:0.1:5;
p1 = normpdf(xs, mu1, s1);
p2 = normpdf(xs, mu2, s2);
plot(xs, pi1*p1, 'r-');
hold on
plot(xs, pi2*p2, 'g-');
grid on
if s1==s2
  syms x % use symbolic math toolbox
  b  = double(solve(pi1*normpdf(x,mu1,s1) - pi2*normpdf(x,mu2,s2)));
  pmax = max(max(p1), max(p2));
  h = line([b b], [0 pmax]); set(h, 'color', 'k', 'linewidth', 3);
else
  b = 0;
end
title(sprintf('%s=%2.1f, %s=%2.1f, %s=%2.1f, %s=%2.1f, %s=%2.1f', ...
	      '\mu_1', mu1, '\mu_2', mu2, '\pi_1', pi1, ...
	      '\sigma_1', s1, '\sigma_2', s2));

