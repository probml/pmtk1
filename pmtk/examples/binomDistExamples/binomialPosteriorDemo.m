%% Posterior Example

data(1).a = 2;      data(2).a = 5;
data(1).b = 2;      data(2).b = 2;
data(1).N1 = 3;     data(2).N1 = 11;
data(1).N0 = 17;    data(2).N0 = 13;

for i = 1:numel(data)
    a = data(i).a;
    b = data(i).b;
    N0 = data(i).N0;
    N1 = data(i).N1;
    N = N1+N0;
    m = BinomDist(N, BetaDist(a,b));
    %m = BernoulliDist(BetaDist(a,b));
    prior = m.mu; % BetaDist
    m = fit(m, 'suffStat', [N1 N],'method','bayesian');
    post = m.mu; % BetaDist
    % The likelihood is the prior with a flat prior
    m2 = BinomDist(N, BetaDist(1,1));
    m2 = fit(m2, 'suffStat', [N1 N],'method','bayesian');
    lik = m2.mu; % BetaDist
    figure;
    h = plot(prior, 'plotArgs', {'r-', 'linewidth', 3});
    legendstr{1} = sprintf('prior Be(%2.1f, %2.1f)', prior.a, prior.b);
    hold on
    h = plot(lik, 'plotArgs', {'k:', 'linewidth', 3});
    legendstr{2} = sprintf('lik Be(%2.1f, %2.1f)', lik.a, lik.b);
    h = plot(post, 'plotArgs', {'b-.', 'linewidth', 3});
    legendstr{3} = sprintf('post Be(%2.1f, %2.1f)', post.a, post.b);
    legend(legendstr)
end

