%% MCMC Sampling from a 2d Gaussians
% We use a N(0, sigma*eye(2)) proposal and see the effect of changing sigma
% We also compare to Gibbs sampling

Sigma = [1 -0.5; -0.5 1];
mu = [1; 1];
m = MvnDist(mu, Sigma);
for i=1:2
  margExact{i} = marginal(m, i);
end
targetFn = @(x) logprob(m,x,false); % unnormalized distribution
fc = makeFullConditionals(m);

N = 500;
xinit = mvnrnd(mu, Sigma);
mcmc{1} = McmcDist('method', 'gibbs', 'fullcond', fc,  'xinit', xinit, 'Nsamples', N);
mcmc{2} = McmcDist('method', 'metrop', 'target', targetFn, 'xinit', xinit, ...
  'Nsamples', N, 'proposal',  @(x) mvnrnd(x, 1*eye(2)));
mcmc{3} = McmcDist('method', 'metrop', 'target', targetFn, 'xinit', xinit, ...
  'Nsamples', N, 'proposal',  @(x) mvnrnd(x, 0.01*eye(2)));
     
names= {'gibbs', 'mh I', 'mh 0.01 I'};

for j=1:length(mcmc)
    ms = mcmc{j};
    ttl = names{j};
    figure;
    plot(m, 'useContour', 'true');
    hold on
    plot(ms);
    title(ttl)
    
    figure;
    for i=1:2
      margApprox{i} = marginal(ms,i);
      subplot2(2,2,i,1);
      [h, histArea] = plot(margApprox{i}, 'useHisto', true);
      hold on
      [h, p] = plot(margExact{i}, 'scaleFactor', histArea, ...
        'plotArgs', {'linewidth', 2, 'color', 'r'});
      title(sprintf('exact m=%5.3f, v=%5.3f', mean(margExact{i}), var(margExact{i})));
      subplot2(2,2,i,2);
      plot(margApprox{i}, 'useHisto', false);
      title(sprintf('approx m=%5.3f, v=%5.3f', mean(margApprox{i}), var(margApprox{i})));
    end
    suptitle(ttl);
    
    figure;
    X = ms.samples;
    for i=1:2
      subplot(1,2,i);
      stem(acf(X(:,i), 30));
      title(ttl)
    end
end



