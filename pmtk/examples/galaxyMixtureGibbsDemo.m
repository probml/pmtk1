%% Gibbs sampling from a Galaxy data fitted to a mixture of six normal distributions.
  %#author Cody Severinski
%#broken

  doPlot = true;
  doPrint = false;

  collapsed = false;
  Nsamples = 20000;
  Nburnin = 1000;

  setSeed(0);
  % Set the number of clusters K
  % the number of observations n to generate in d dimensions,
  % and the true mu, Sigma for these
  K = 6;
  load('galaxies.csv'); [n,d] = size(galaxies);
  % scale the data in units of 1000
  galaxies = galaxies / 1000;
  galaxies = galaxies(randperm(n));
  % specify the prior distribution to use.
  chosenPrior = MvnInvWishartDist('mu', mean(galaxies)', 'Sigma', diag(var((galaxies))) / K^(2/d), 'dof', d + 2, 'k', 0.01);
 % model = MvnDist('distributions',copy( MvnDist('mu', zeros(d,1),'Sigma', diag(ones(d,1)), 'prior', chosenPrior), K,1) ) ;
model = MixMvnGibbs('distributions',copy( MvnDist('-mu', zeros(d,1),'-Sigma', diag(ones(d,1)), '-prior', chosenPrior), K,1) ) ;

  % Set the prior distribution on the mixing weights to be Dirichlet(1,..., 1)
  model.mixingWeights.prior = DirichletDist(ones(K,1));

  % Initiate the sampler
  if(collapsed)
    [muDist, SigmaDist, latentDist, mixDist] = collapsedGibbs(model,galaxies,'Nsamples', Nsamples, 'Nburnin', Nburnin, 'thin', 1);
  else
    [muDist, SigmaDist, latentDist, mixDist] = latentGibbsSample(model,galaxies,'Nsamples', Nsamples, 'Nburnin', Nburnin, 'thin', 1, 'verbose', true);
  end

  % Perform postprocessing on the labels
  [muoutDist, SigmaoutDist, latentoutDist, mixoutDist, permOut] = processLabelSwitch(model,latentDist, muDist, SigmaDist, mixDist, galaxies);

  


  %% Now make some pretty pictures
%{
  [mu, sigma] = getMuSigma(mcmc);
  den = getDensities(mcmcout);

  [muout, sigmaout] = getMuSigma(mcmcout);
  denout = getDensities(mcmcout);

  traceplot(mu, 'galaxy_muVsItr'); 
  traceplot(muout, 'galaxy_muVsItr_post')

  plotDensities(den, 'galaxy_densityEst');
  plotDensities(denout, 'galaxy_densityEst_post');
  function [mu,sigma] = getMuSigma(mcmc, plottitle)
    Nitr = size(mcmc.param,2);
    mu = zeros(K,Nitr); sigma = zeros(K,Nitr);
    for itr=1:Nitr
      for k=1:K
        mu(k,itr) = mcmc.param{k,itr}.mu;
        sigma(k,itr) = mcmc.param{k,itr}.Sigma;
      end
    end
  end

  function [den] = getDensities(den, plottitle)
    Nitr = size(mcmc.param,2);
    x = 0:0.01:40;
    den = zeros(K,length(x));
    for itr=1:Nitr
      for k=1:K
        den(k,:) = den(k,:) + normpdf(x,mu(k,itr), sqrt(sigma(k,itr)));
      end
    end

    den = den / Nitr;
  end

  function [] = traceplot(mu, plottitle)
    figure(); subplot(3,2,1);
    for k=1:K
      subplot(3,2,k);
      plot(mu(k,:));
      ylabel(sprintf('Mean, Cluster %d', k));
    end
    if(doPrint)
      pdfcrop; print_pdf(sprintf(plottitle));
    end
  end

  function [] = plotDensities(den, plottitle)
    figure(); subplot(3,2,1);
    x = 0:0.01:40;
    for k=1:K
      subplot(3,2,k);
      plot(x,den(k,:),'Linewidth',3);
      xlabel(sprintf('Density estimate for mean of cluster %d', k));
    end
    if(doPrint)
      pdfcrop; print_pdf(sprintf(plottitle));
    end
  end
%}

