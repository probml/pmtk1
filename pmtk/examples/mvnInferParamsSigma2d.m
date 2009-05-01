%% Demo of inferrring 2d MVN covariance given fixed mean 
setSeed(0);
muTrue = [0 0]'; Ctrue = 0.1*[2 1; 1 1];
mtrue = MvnDist(muTrue, Ctrue);
xrange = 2*[-1 1 -1 1];
n = 20;
X = sample(mtrue, n);
ns = [10];
figure;
useContour = true;
plot(X(:,1), X(:,2), '.', 'markersize',15);
axis(xrange);
hold on
%plotContour2d(mtrue);
gaussPlot2d(mtrue.mu, mtrue.Sigma);
title('truth'); grid on;

%prior = invWishartDist(10, Ctrue); % cheat!
prior = InvWishartDist(2, eye(2));
plotMarginals(prior);
%set(gcf, 'name', 'prior');
suplabel('prior');
if doPrintPmtk, doPrintPmtkFigures('MVNcovDemoPriorMarg'); end;

plotSamples2d(prior, 9);
subplot(3,3,1); gaussPlot2d(mtrue.mu, mtrue.Sigma);  title('truth');
suplabel('prior');
if doPrintPmtk, doPrintPmtkFigures('MVNcovDemoPriorSamples'); end;

for i=1:length(ns)
    n = ns(i);
    m = Mvn_InvWishartDist(muTrue, prior);
    m = fit(m, 'data', X(1:n,:));
    post = m.SigmaDist;
    plotMarginals(post);
    suplabel(sprintf('post after %d obs', n));
    if doPrintPmtk, doPrintPmtkFigures(sprintf('MVNcovDemoPost%dMarg', n)); end;

    plotSamples2d(post, 9);
    hold off
    subplot(3,3,1); gaussPlot2d(mtrue.mu, mtrue.Sigma); title('truth');
    suplabel(sprintf('post after %d obs', n));
    if doPrintPmtk, doPrintPmtkFigures(sprintf('MVNcovDemoPost%dSamples', n)); end;
end