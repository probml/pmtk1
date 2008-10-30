%% MVN Infer Mean 1D
priorVars = [1 5];
for i=1:numel(priorVars)
    priorVar = priorVars(i);
    prior = mvnDist(0, priorVar);
    sigma2 = 1;
    m = mvnDist(prior, sigma2);
    x = 3;
    m = inferParams(m, 'data', x);
    post = m.mu;
    % The likelihood is proportional to the posterior when we use a flat prior
    priorBroad = mvnDist(0, 1e10);
    m2 = mvnDist(priorBroad, sigma2);
    m2 = inferParams(m2, 'data', x);
    lik = m2.mu;
    % Now plot
    figure;
    xrange = [-5 5];
    hold on
    plot(prior, 'xrange', xrange, 'plotArgs', { 'r-', 'linewidth', 2});
    legendstr{1} = 'prior';
    plot(lik, 'xrange', xrange,'plotArgs', {'k:o', 'linewidth', 2});
    legendstr{2} = 'lik';
    plot(post, 'xrange', xrange,'plotArgs', {'b-.', 'linewidth', 2});
    legendstr{3} = 'post';
    legend(legendstr)
    title(sprintf('prior variance = %3.2f', priorVar))
end

