
% Johnson and Albert p35

initVal = [2 5]; 

[mu, C, logZ] = laplaceApprox(@smokingCancerLogpost, initVal)


% Credible interval for alpha
z = norminv(1-0.025);
sigma = sqrt(C(1,1));
alphaHat = mu(1);
credibleInterval = [alphaHat - z*sigma, alphaHat + z*sigma]
credibleInterval = [norminv(0.025, alphaHat, sigma), norminv(1-0.025, alphaHat, sigma)]

ppos = normcdf(alphaHat/sigma) % Prob(alpha>0)

% Plot exact posterior
[xs, ys] = meshgrid(-1:0.1:6, 2:0.1:10);
xrange = xs(1,:);
yrange = ys(:,1);
xy = [xs(:) ys(:)];
p = exp(smokingCancerLogpost(xy));
p = reshape(p, size(xs));
h = max(p(:));
vals = [0.1*h 0.01*h 0.001*h];
figure; 
contour(xrange, yrange, p); %, vals);
title('exact')
axis('square')
grid on
hold on
plot(mu(1), mu(2), 'x', 'markersize', 12)
title('unnormalized posterior')

figure; imagesc(p); axis xy
xt = get(gca,'xtick');
set(gca, 'xticklabel', xrange(xt));
yt = get(gca,'ytick');
set(gca, 'yticklabel', yrange(yt));

p2 = mvnpdf(xy, mu(:)', C);
p2 = reshape(p2, size(xs));
h2 = max(p2(:));
vals2 = [0.1*h2 0.01*h2 0.001*h2];
figure; 
contour(xrange, yrange, p2); %, vals2)
title('Laplace approximation')
axis('square')
grid on
hold on
plot(mu(1), mu(2), 'x', 'markersize', 12)

