% ppcademo
% Based on figure 7.6 of the netlab book

seed = 0; randn('state', seed);
n = 5;
X=[randn(n,2)+2.*ones(n,2);2.*randn(n,2)-2.*ones(n,2)];
[n d] = size(X);
[W, mu, sigma2] = ppcaFit(X, 1);
[Z, postCov] = ppcaPost(X, W, mu, sigma2);
Xrecon = Z*W' + repmat(mu, n,1);
figure(2);clf;
plot(mu(1), mu(2), '*', 'markersize', 15, 'color', 'r');
hold on
plot(X(:,1), X(:,2), 'o');
hold on
plot(Xrecon(:,1), Xrecon(:,2), 'x');
for i=1:n
  line([Xrecon(i,1) X(i,1)], [Xrecon(i,2) X(i,2)])
end
% plot the linear subspace
Z2 = [-2;1.5];
Xrecon2 = Z2*W' + repmat(mu, 2,1);
line([Xrecon2(1,1) Xrecon2(2,1)], [Xrecon2(1,2) Xrecon2(2,2)])
axis square
