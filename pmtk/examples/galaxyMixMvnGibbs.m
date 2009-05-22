%% Gibbs sampling from a Galaxy data fitted to a mixture of six normal distributions.
%#author Cody Severinski

  doPlot = true;
  doPrint = false;
  setSeed(0);
  K = 6;
  load('galaxies.csv'); [n,d] = size(galaxies);
  % scale the data in units of 1000, then permute
  galaxies = galaxies / 1000; galaxies = galaxies(randperm(n));

  % specify the prior distribution to use.
% This is the MvnInvWishart equivalent parametrization suggested by
%  @article{fearnhead2004pfm,
%    title={{Particle filters for mixture models with an unknown number of components}},
%    author={Fearnhead, P.},
%    journal={Statistics and Computing},
%    volume={14},
%    number={1},
%    pages={11--21},
%    year={2004},
%    publisher={Springer}
%  }
  chosenPrior = MvnInvWishartDist('mu', 20, 'Sigma', 4, 'dof', 4, 'k', 1/15^2);
  model = MixMvnGibbs('distributions',copy( MvnDist('-mu', zeros(d,1),'-Sigma', diag(ones(d,1)), '-prior', chosenPrior), K,1) ) ;
  model.mixingDistrib.prior = DirichletDist(ones(K,1));
  % Initiate the sampler
  method = {'full', 'collapsed'};
  for m=1:length(method)
    [fitted{m}, latent{m}] = fit(model, galaxies, '-method', method{m}, '-Nsamples', 1250, '-Nburnin', 250, '-verbose', true);
  end

  for m=1:length(method)
    traceplot(fitted{m}); suptitle(sprintf('%s Gibbs sampling', method{m}));
    if doPrintPmtk, printPmtkFigures(sprintf('galaxyMixMvn%sTraceplot', method{m})); end;
    convergencePlot(fitted{m}, galaxies); suptitle(sprintf('%s Gibbs sampling', method{m}));
    if doPrintPmtk, printPmtkFigures(sprintf('galaxyMixMvn%sConvergence', method{m})); end;
  end
