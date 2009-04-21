%% MH Sampling from a mixture of two 1d Gaussians using a 1d Gaussian proposal
%#broken
%m = MixGaussDist('K', 2, 'mu', [-50 50], 'Sigma', reshape([10^2 10^2], [1 1 2]), ...
  %'mixweights', [0.3 0.7]);
  
m = MvnMixDist('distributions',{MvnDist(-50,100),MvnDist(50,100)},'mixingWeights',DiscreteDist([0.3;0.7]));
  
targetFn = @(x) (logprob(m,x));
    
% Cool plot from Christoph Andrieu
sigmas = [10 100 500];
for i=1:length(sigmas)
    sigma_prop = sigmas(i);
    setSeed(0); 
    proposalFn = @(x) (x + (sigma_prop * randn(1,1)));
    N = 1000;
    %xinit  = m.mu(2) + randn(1,1);
    xinit  = subd(marginal(m,2),'mu') + randn(1,1);
    [x, ar] = mhSample('target', targetFn, 'proposal', proposalFn, ...
      'xinit', xinit, 'Nsamples', N);
    figure;
    nb_iter = N;
    x_real = linspace(-100, 100, nb_iter);
    y_real = exp(logprob(m, x_real(:)));
    Nbins = 100;
    plot3(1:nb_iter, x, zeros(nb_iter, 1))
    hold on
    plot3(ones(nb_iter, 1), x_real, y_real)
    [u,v] = hist(x, linspace(-100, 100, Nbins));
    plot3(zeros(Nbins, 1), v, u/nb_iter*Nbins/200, 'r')
    hold off
    grid
    view(60, 60)
    xlabel('Iterations')
    ylabel('Samples')
    title(sprintf('MH with N(0,%5.3f^2) proposal', sigma_prop))
end
drawnow

% Convergence diagnosistics 
seeds = 1:3;
nseeds = length(seeds);
N = 1000;
X = zeros(N, nseeds);
for s=1:length(sigmas)
  sigma_prop = sigmas(s);
  proposalFn = @(x) (x + (sigma_prop * randn(1,1)));
  for i=1:length(seeds)
    setSeed(seeds(i));
    %xinit  = m.mu(2) + randn(1,1);
    xinit  = subd(marginal(m,2),'mu') + randn(1,1);
    [X(:,i), ar] = mhSample('target', targetFn, 'proposal', proposalFn, ...
      'xinit', xinit, 'Nsamples', N);
  end
  plotConvDiagnostics(X, sprintf('sigma prop %5.3f', sigmas(s)));
end


