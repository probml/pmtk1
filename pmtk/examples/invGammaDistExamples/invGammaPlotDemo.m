%% Plot Demo

small = false
if small
    as = [0.01 0.1 1];
    bs = as;
    xr = [0 2];
else
    as = [0.1 0.5 1 2];
    bs = 1*ones(1,length(as));
    xr = [0 5];
end
figure;
[styles, colors, symbols] = plotColors;
for i=1:length(as)
    a = as(i); b = bs(i);
    plot(invGammaDist(a,b), 'xrange', xr, 'plotArgs', {styles{i}, 'linewidth', 2});
    hold on
    legendStr{i} = sprintf('a=%4.3f,b=%4.3f', a, b);
end
legend(legendStr);
title('InvGamma distributions')