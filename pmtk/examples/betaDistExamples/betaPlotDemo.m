%% Plotting Beta Distributions
as = [0.1 1 2 8]; bs = [0.1 1 3 4];
figure;
[styles, colors, symbols] = plotColors;
for i=1:length(as)
    a = as(i); b = bs(i);
    plot(betaDist(a,b), 'plotArgs', {styles{i}, 'linewidth', 2});
    hold on
    legendStr{i} = sprintf('a=%2.1f,b=%2.1f', a, b);
end
legend(legendStr);
title('beta distributions')