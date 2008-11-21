function laplaceDemoJohnsonSmoking()
% Johnson and Albert p35

initVal = [2 5]; 

%{
  function [f,g,H] = foo(theta)
    fn = @(theta) johnsonSmokingLogpostT(theta);
    f = fn(theta);
    %g = numericalGradient(@fn, theta, {});
    %H = numericalHessian(@fn, theta, {});
    g = gradest(fn, theta)';
    H = hessian(fn, theta);
  end

[f,g,H] = foo(initVal);
%}

disp('johnson')
[mu, C, logZ] = laplaceApproxJohnson(@johnsonSmokingLogpostT, initVal)

disp('minfunc')
[mu2, C2, logZ2] = laplaceApprox(@johnsonSmokingLogpostT, initVal)

keyboard

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

p2 = mvnpdf(xy', mu', C);
p2 = reshape(p2, size(alphas));
h2 = max(p2(:));
vals2 = [0.1*h2 0.01*h2 0.001*h2];
figure(3); clf; 
contour(alphas(1,:), etas(:,1), p2, vals2)
title('Laplace')
axis('square')
grid on
hold on
plot(mu(1), mu(2), 'x', 'markersize', 12)

keyboard



end