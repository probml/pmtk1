%% MVN Infer Parameters
doSave = false;
folder = 'C:\kmurphy\PML\pdfFigures';
seed = 0; randn('state', seed); rand('twister', seed);
muTrue = [0 0]'; Ctrue = 0.1*[2 1; 1 1];
mtrue = MvnDist(muTrue, Ctrue);
xrange = 2*[-1 1 -1 1];
n = 20;
X = sample(mtrue, n);
ns = [20];
figure;
useContour = true;
plot(X(:,1), X(:,2), '.', 'markersize',15);
axis(xrange);
hold on
%plotContour2d(mtrue);
gaussPlot2d(mtrue.mu, mtrue.Sigma);
title('truth'); grid on;
fname = fullfile(folder, sprintf('MVNcovDemoData.pdf'));
if doSave, pdfcrop; print(gcf, '-dpdf', fname); end

%prior = invWishartDist(10, Ctrue); % cheat!
prior = InvWishartDist(2, eye(2));
plotMarginals(prior);
%set(gcf, 'name', 'prior');
suplabel('prior');
fname = fullfile(folder, sprintf('MVNcovDemoPriorMarg.pdf'));
if doSave, pdfcrop; print(gcf, '-dpdf', fname); end

plotSamples2d(prior, 9);
subplot(3,3,1); gaussPlot2d(mtrue.mu, mtrue.Sigma);  title('truth');
suplabel('prior');
fname = fullfile(folder, sprintf('MVNcovDemoPriorSamples.pdf'));
if doSave, pdfcrop; print(gcf, '-dpdf', fname); end

for i=1:length(ns)
    n = ns(i);
    m = MvnDist(muTrue, prior);
    m = fit(m, 'data', X(1:n,:));
    post = m.Sigma;
    plotMarginals(post);
    suplabel(sprintf('post after %d obs', n));
    fname = fullfile(folder, sprintf('MVNcovDemoPost%dMarg.pdf', n));
    if doSave, pdfcrop; print(gcf, '-dpdf', fname); end

    plotSamples2d(post, 9);
    subplot(3,3,1); gaussPlot2d(mtrue.mu, mtrue.Sigma); title('truth');
    suplabel(sprintf('post after %d obs', n));
    fname = fullfile(folder, sprintf('MVNcovDemoPost%dSamples.pdf', n));
    if doSave, pdfcrop; print(gcf, '-dpdf', fname); end
end