%% Sequential Bayesian Updating. 
setSeed(0);
m = bernoulliDist(0.7);
n = 100;
X = sample(m, n);
figure; hold on;
[styles, colors, symbols] =  plotColors();
ns = [0 5 50 100];
for i=1:length(ns)
    n = ns(i);
    mm = bernoulliDist(betaDist(1,1));
    mm = inferParams(mm, 'data', X(1:n));
    plot(mm.mu, 'plotArgs', {styles{i}, 'linewidth', 2});
    legendstr{i} = sprintf('n=%d', n);
end
legend(legendstr,'Location','NorthWest');
xbar = mean(X);
pmax = 10;
h=line([xbar xbar], [0 pmax]); set(h, 'linewidth', 3);