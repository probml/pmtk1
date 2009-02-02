%% MH Sampling from a 2d Gaussians
% We use a N(0, sigma*eye(2)) proposal and see the effect of changing sigma
cls;
Sigma = [1 -0.5; -0.5 1];
mu = [1; 1];
N = 500; 

% In this code, we call mhSample directly,
% and do not use the PMTK object system
% See mcmcMvn2d for how to use mcmc inside the object system.

targetFn = @(x) log(gausspdfUnnormalized(x, mu, Sigma));
xinit = mvnrnd(mu, Sigma);
S{1} = mhSample('target', targetFn, 'xinit', xinit, ...
  'Nsamples', N, 'proposal',  @(x) mvnrnd(x, 1*eye(2)));
S{2} = mhSample('target', targetFn, 'xinit', xinit, ...
  'Nsamples', N, 'proposal',  @(x) mvnrnd(x, 0.01*eye(2)));

names= {'mh I', 'mh 0.01 I'};

for j=1:length(S)
    ms = S{j};
    ttl = names{j};
    figure;
    gaussPlot2d(mu, Sigma);
    hold on
    plot(ms(:,1), ms(:,2), '.');
    title(ttl)
    
    figure;
    for i=1:2
      subplot(1,2,i);
      stem(acf(ms(:,i), 30));
      title(ttl)
    end
end



