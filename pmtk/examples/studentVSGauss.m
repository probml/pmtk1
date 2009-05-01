%% Student T vs Gaussian
%#testPMTK
useLog = false;
dofs = [0.1 1 5];
xr = [-4 4];
figure;
[styles, colors, symbols] = plotColors;
N = length(dofs);
for i=1:N
    dof = dofs(i);
    h = plot(StudentDist(dof, 0, 1), 'useLog', useLog, ...
        'xrange', xr, 'plotArgs', {styles{i},'linewidth',2});
    %set(h,'color',colors(i)); set(h,'marker',symbols(i))
    hold on
    legendStr{i} = sprintf('t(%s=%2.1f)', '\nu', dof);
end
h = plot(GaussDist(0, 1), 'useLog', useLog, ...
    'xrange', xr, 'plotArgs', {styles{N+1},'linewidth',2});
legendStr{end+1} = 'N(0,1)';
legend(legendStr)
if useLog, ylabel('log density'); else ylabel('density'); end
if(useLog)
if doPrintPmtk, doPrintPmtkFigures('studentTvsGaussLog'); end;
else
if doPrintPmtk, doPrintPmtkFigures('studentTvsGauss'); end;
end