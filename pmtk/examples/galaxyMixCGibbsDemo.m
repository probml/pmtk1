%% Collapsed Gibbs sampling from a Galaxy data fitted to a mixture of six normal distributions
%#testPMTK
%#author Cody Severinski

doPlot = true;
doPrint = false;
setSeed(0);
% Set the number of clusters K
% the number of observations n to generate in d dimensions,
% and the true mu, Sigma for these
K = 6;
load('galaxies.csv'); [n,d] = size(galaxies);
% scale the data in units of 10000
galaxies = galaxies / 1000;
galaxies = galaxies(randperm(n));
% specify the prior distribution to use.  If we simply use 'niw', then the prior will change within the sampler.  This is bad...
chosenPrior = MvnInvWishartDist('mu', mean(galaxies), 'Sigma', diag(var((galaxies))), 'dof', d + 1, 'k', 0.001);
model = MvnMixDist('distributions',copy( MvnDist(zeros(d,1),diag(ones(d,1)), 'prior', chosenPrior), K,1) ) ;

% Set the prior distribution on the mixing weights to be Dirichlet(1,..., 1)
model.mixingWeights.prior = DirichletDist(ones(K,1));

Nsamples = 5000; Nburnin = 500;
% Initiate the sampler
%profile on;
mcmc = collapsedGibbs(model,galaxies,'Nsamples', Nsamples, 'Nburnin', Nburnin, 'thin', 1);
%profile viewer; profile off;
% Perform postprocessing on the labels - not working yet
[mcmcout,permout] = processLabelSwitch(model,mcmc,galaxies);


% extract and display the results of the mu for the clusters -- This will demonstrate the label switching problem
if(doPlot)
  Nitr = size(mcmc.param,2);
  mu = zeros(K,Nitr); sigma = zeros(K,Nitr);
  for itr=1:Nitr
    for k=1:K
      mu(k,itr) = mcmc.param{k,itr}.mu;
      sigma(k,itr) = mcmc.param{k,itr}.Sigma;
    end
  end

  figure(); subplot(3,2,1);
  for k=1:K
    subplot(3,2,k);
    plot(mu(k,:));
    ylabel('Mean, Cluster 1');
    %axis([0,Nitr, 5, 30]);
    %ylabel(sprintf('%s',text('Interpreter','latex', 'String', '$\mu_k$')));
  end
end
if(doPrint)
  pdfcrop; print_pdf('galaxy_muVsItr');
end

% Density estimation
if(doPlot)
  x = 0:0.01:40;
  den = zeros(K,length(x));
  for itr=1:Nitr
    for k=1:K
      den(k,:) = den(k,:) + normpdf(x,mu(k,itr), sqrt(sigma(k,itr)));
    end
  end
  den = den / Nitr;
  

  figure(); subplot(3,2,1);
  for k=1:K
    subplot(3,2,k);
    plot(x,den(k,:),'Linewidth',3);
    xlabel(sprintf('Density estimate for mean of cluster %d', k));
  end
end
if(doPrint)
  pdfcrop; print_pdf('galaxy_densityEst');
end


%% Now redo the density estimates using the permuted parameters
% extract and display the results of the mu for the clusters
if(doPlot)
  Nitr = size(mcmcout.param,2);
  muout = zeros(K,Nitr); sigmaout = zeros(K,Nitr);
  for itr=1:Nitr
    for k=1:K
      muout(k,itr) = mcmcout.param{k,itr}.mu;
      sigmaout(k,itr) = mcmcout.param{k,itr}.Sigma;
    end
  end

  figure(); subplot(3,2,1);
  for k=1:K
    subplot(3,2,k);
    plot(muout(k,:));
    ylabel('Mean, Cluster 1');
    %ylabel(sprintf('%s',text('Interpreter','latex', 'String', '$\mu_k$')));
  end
end
if(doPrint)
  pdfcrop; print_pdf('galaxy_muVsItr_post');
end

% Density estimation
if(doPlot)
  x = 0:0.01:40;
  denout = zeros(K,length(x));
  for itr=1:Nitr
    for k=1:K
      denout(k,:) = denout(k,:) + normpdf(x,muout(k,itr), sqrt(sigmaout(k,itr)));
    end
  end

  denout = denout/Nitr;

  figure(); subplot(3,2,1);
  for k=1:K
    subplot(3,2,k);
    plot(x,denout(k,:),'Linewidth',3);
    xlabel(sprintf('Density estimate for mean of cluster %d', k));
  end
end
if(doPlot)
  pdfcrop; print_pdf('galaxy_densityEst_post');
end



