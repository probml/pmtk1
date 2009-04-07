function fisherLDAdemo()

doPlot = 1;
folder = 'C:\kmurphy\PML\Figures';

seed = 0; rand('state', seed); randn('state', seed);
mix = gmm(2, 2, 'full');
mix.centres = [1 3;
	       3 1];
%mix.covars(:,:,1) = [4 0.01; 0.01 4];
mix.covars(:,:,1) = [4 0; 0 0.1];
mix.covars(:,:,2) = mix.covars(:,:,1);
[X,Y] = gmmsamp(mix, 200);

maleNdx = find(Y == 1);
femaleNdx = find(Y == 2);

figure(1);clf;
plot(X(maleNdx,1), X(maleNdx,2), 'bx');
hold on
plot(X(femaleNdx,1), X(femaleNdx,2), 'ro');
axis equal

muMale = mean(X(maleNdx,:));
muFemale = mean(X(femaleNdx,:));
plot(muMale(1), muMale(2), 'bx','markerSize',15,'linewidth',3)
plot(muFemale(1), muFemale(2), 'ro','markerSize',15,'linewidth',3)
mu = mean(X);
plot(mu(1), mu(2), 'k*','markerSize',15,'linewidth',3)


wMean = (muMale - muFemale)';
wFisher = fisherLDA(X,Y);
[wPCA] = pcaMLABA(X, 1);

%f = sprintf('0=%g*x + %g*y', w(1), w(2));
%h = ezplot(f, [-2 2 -2 2]);
%h = ezplot(@(x,y) w(1)*x+w(2)*y, [-2 2 -2 2]);
%set(h, 'color', 'r', 'linewidth', 2);

h1=line([muMale(1) muFemale(1)], [muMale(2) muFemale(2)]);
set(h1,'linewidth',2, 'color', 'k');

s = 0.2;
h2=line([mu(1)-s*wFisher(1) mu(1)+s*wFisher(1)], [mu(2)-s*wFisher(2) mu(2)+s*wFisher(2)]);
set(h2, 'color', 'r', 'linewidth', 3, 'linestyle', ':')

s  = 3;
h3=line([mu(1)-s*wPCA(1) mu(1)+s*wPCA(1)], [mu(2)-s*wPCA(2) mu(2)+s*wPCA(2)]);
set(h3, 'color', 'g', 'linewidth', 3, 'linestyle', '--')

str = {'means', 'fisher', 'pca'};
legend([h1 h2 h3],str)

fname = sprintf('%s/fisherLDAdemoData.eps', folder)
if doPlot, print(gcf, '-depsc', fname); end

XprojMean = X*wMean;
XprojFisher = X*wFisher;
XprojPCA = X*wPCA;

figure(2);clf; plotData(XprojMean, maleNdx, femaleNdx, str{1})
fname = sprintf('%s/fisherLDAdemoProjMean.eps', folder)
if doPlot, print(gcf, '-depsc', fname); end

figure(3);clf; plotData(XprojFisher, maleNdx, femaleNdx, str{2})
fname = sprintf('%s/fisherLDAdemoProjFisher.eps', folder)
if doPlot, print(gcf, '-depsc', fname); end

figure(4);clf; plotData(XprojPCA, maleNdx, femaleNdx, str{3})
fname = sprintf('%s/fisherLDAdemoProjPCA.eps', folder)
if doPlot, print(gcf, '-depsc', fname); end


%%%%%%%%

function plotData(Xproj, maleNdx, femaleNdx, ttl)

[nMale, xMale]  = hist(Xproj(maleNdx));
bar(xMale, nMale, 'b')
[nFemale, xFemale] = hist(Xproj(femaleNdx));
hold on
bar(xFemale, nFemale, 'r')
title(ttl)
