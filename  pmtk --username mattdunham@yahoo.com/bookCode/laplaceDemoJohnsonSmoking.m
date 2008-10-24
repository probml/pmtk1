function laplaceDemoJohnsonSmoking()
% Johnson and Albert p35

initVal = [2; 5]; 
grad = @(theta) numericalGradient(@johnsonSmokingLogpost, theta, {});
hess = @(theta) numericalHessian(@johnsonSmokingLogpost, theta, {});
[mode, C, logZ, niter] = laplaceApprox(initVal, @johnsonSmokingLogpost, grad, hess)

% Credible interval for alpha
z = norminv(1-0.025);
sigma = sqrt(C(1,1));
alphaHat = mode(1);
credibleInterval = [alphaHat - z*sigma, alphaHat + z*sigma]
credibleInterval = [norminv(0.025, alphaHat, sigma), norminv(1-0.025, alphaHat, sigma)]

ppos = normcdf(alphaHat/sigma) % Prob(alpha>0)

% Plot exact and approximate posterior
[alphas, etas] = meshgrid(-1:0.1:6, 2:0.1:10);
xy = [alphas(:)'; etas(:)'];
p = exp(johnsonSmokingLogpost(xy));
p = reshape(p, size(alphas));
h = max(p(:));
vals = [0.1*h 0.01*h 0.001*h];
figure(1); clf; 
contour(alphas(1,:), etas(:,1), p, vals);
title('exact')
axis('square')
grid on
hold on
plot(mode(1), mode(2), 'x', 'markersize', 12)

p2 = mvnpdf(xy', mode', C);
p2 = reshape(p2, size(alphas));
h2 = max(p2(:));
vals2 = [0.1*h2 0.01*h2 0.001*h2];
figure(3); clf; 
contour(alphas(1,:), etas(:,1), p2, vals2)
title('Laplace')
axis('square')
grid on
hold on
plot(mode(1), mode(2), 'x', 'markersize', 12)
