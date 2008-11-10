%% MVN Infer Mean 2D Demo
setSeed(0);
muTrue = [0.5 0.5]'; Ctrue = 0.1*[2 1; 1 1];
mtrue = MvnDist(muTrue, Ctrue);
xrange = [-1 1 -1 1];
n = 10;
X = sample(mtrue, n);
ns = [2 5 10];
figure;
useContour = true;
nr = 2; nc = 3;
subplot(nr,nc,1);
plot(X(:,1), X(:,2), '.', 'markersize',15);
%hold on; for i=1:n, text(X(i,1), X(i,2), sprintf('%d', i)); end
axis(xrange); title('data'); grid on; axis square
subplot(nr,nc,2);
plot(mtrue, 'xrange', [-1 2 -1 2], 'useContour', useContour);
%gaussPlot2d(mtrue.mu, mtrue.Sigma);
title('truth'); grid on; axis square
prior = MvnDist([0 0]', 0.1*eye(2));
subplot(nr,nc,3); plot(prior, 'xrange', xrange, 'useContour', useContour);
title('prior'); grid on; axis square
for i=1:length(ns)
    n = ns(i);
    m = MvnDist(prior, Ctrue);
    m = inferParams(m, 'data', X(1:n,:));
    post = m.mu;
    subplot(nr,nc,i+3); plot(post, 'xrange', xrange, 'useContour', useContour, 'npoints', 150);
    title(sprintf('post after %d obs', n)); grid on; axis square
end