%% Plot a Gamma Distribution
%as = [1 1.5 2  1 1.5 2]; bs = [1 1 1 1.5 1.5 1.5];
%#testPMTK
as = [1 1.5 2];
b = 1;
bs = b*ones(1,length(as));
figure;
[styles, colors, symbols] = plotColors;
for i=1:length(as)
    a = as(i); b = bs(i);
    plot(GammaDist(a,b), 'xrange', [0 7], 'plotArgs', {styles{i}, 'linewidth', 2});
    hold on
    legendStr{i} = sprintf('a=%2.1f,b=%2.1f', a, b);
end
legend(legendStr);
title('Gamma distributions')

if doPrintPmtk, printPmtkFigures('gammaDist'); end;