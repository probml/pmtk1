%% Sequential Updating of mu
setSeed(1);
mutrue = 5; Ctrue = 10;
mtrue = mvnDist(mutrue, Ctrue);
n = 500;
X = sample(mtrue, n);
ns = [0 2 5 50]
figure; hold on;
pmax = -inf;
[styles, colors, symbols] =  plotColors();
for i=1:length(ns)
    k = 0.001;
    prior = mvnDist(0, 1/k);
    n = ns(i);
    m = inferParams(mvnDist(prior, Ctrue), 'data', X(1:n));
    post = m.mu;
    [h(i), p]= plot(post, 'plotArgs', {styles{i}, 'linewidth', 2}, 'xrange', [0 10]);
    legendstr{i} = sprintf('n=%d', n);
    pmax = max(pmax, max(p));
    xbar = mean(X(1:n)); vbar = var(X(1:n));
    %h(i)=line([xbar xbar], [0 pmax],'color',colors(i),'linewidth',3);
end
legend(h,legendstr);
title(sprintf('prior = N(mu0=0, v0=%5.3f), true %s = %5.3f', 1/k, '\mu', mutrue))
line([mutrue, mutrue], [0 pmax],'color','k','linewidth',3);