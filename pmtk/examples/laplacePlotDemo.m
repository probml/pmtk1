%% Plot Laplace Distributions
%#testPMTK
mus = [0 0 0 -5];
bs = [1 2 4 4];
figure;
styles = plotColors;
xr = [-10 10];
for j=1:length(mus)
    obj = LaplaceDist(mus(j), bs(j));
    h=plot(obj, 'plotArgs', styles{j}, 'xrange', xr, 'npoints', 100);
    hold on
    legendStr{j} = sprintf('%s=%3.1f, b=%3.1f', '\mu', mus(j), bs(j));
end
legend(legendStr)
title('Laplace distributions')
if doPrintPmtk, doPrintPmtkFigures('laplace'); end;