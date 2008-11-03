%% Conditioning a 2D MVN
% Take a horizontal slice thru a 2d Gaussian and plot the resulting
% conditional
mu = [0 0]';
rho = 0.5;
%S  = [4 1; 1 1];
S = [1 rho; rho 1];
obj = mvnDist(mu,S);
figure;
gaussPlot2d(obj.mu, obj.Sigma);
hold on;
[U,D] = eig(S); %plot
sf=  -2.5;
line([mu(1) mu(1)+sf*sqrt(D(1,1))*U(1,1)],[mu(2) mu(2)+sf*sqrt(D(1,1))*U(2,1)],'linewidth',2)
line([mu(1) mu(1)+sf*sqrt(D(2,2))*U(1,2)],[mu(2) mu(2)+sf*sqrt(D(2,2))*U(2,2)],'linewidth',2)
%line([-5 5], [-5 5]);
x2 = 1; line([-5 5], [x2 x2],  'color', 'r', 'linewidth', 2);
%post = conditional(obj, 2, x2); % 2 is the y axis
obj = enterEvidence(obj, 2, x2);
post = marginal(obj, 1);
%plot(post, 'xrange', [-4 4]);
xs = -5:0.1:5;
ps = exp(logprob(post, xs));
ps = 50*normalize(ps);
plot(xs, 1+ps, 'k', 'linewidth',2 );
postMu = mean(post);
line([postMu postMu], [-4 4], 'color', 'k', 'linewidth', 2, 'linestyle', '-');
%grid on
title(sprintf('p(x1,x2)=N([0 0], [1 %3.2f; %3.2f 1]), p(x1|x2=%3.1f)=N(x1|%3.2f, %3.2f)', ...
    rho, rho, x2, mean(post), var(post)));