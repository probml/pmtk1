%% Run full gibbs sampling, collapsed gibbs sampling, EM, and VB EM on the old Faithful dataset
% Print the resulting cluster means in a table
%#author Cody Severinski

cls; setSeed(0);
K = 6;
load('galaxies.csv'); [n,d] = size(galaxies);
Nsamples = 1250; Nburnin = 250;
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
mixPrior = DirichletDist(ones(K,1));

method = {'gibbs full', 'gibbs collapsed', 'EM', 'VB EM'};

model{1} = MixMvnGibbs('distributions',copy( MvnDist('-mu', zeros(d,1),'-Sigma', diag(ones(d,1)), '-prior', chosenPrior), K,1) ) ;
model{1}.mixingDistrib.prior = mixPrior;
fitted{1} = fit(model{1}, galaxies, '-method', 'full', '-Nsamples', Nsamples, '-Nburnin', Nburnin, '-verbose', true);
fitted{2} = fit(model{1}, galaxies, '-method', 'collapsed', '-Nsamples', Nsamples, '-Nburnin', Nburnin, '-verbose', true);

model{3} = MixMvn('-nmixtures', K, '-ndims', d) ;
model{3}.distributions = copy( MvnDist('-mu', zeros(d,1),'-Sigma', diag(ones(d,1)), '-prior', chosenPrior), K,1);
model{3}.mixingDistrib.prior = mixPrior;
fitted{3} = fit(model{3}, '-data', galaxies);


model{4} = MixMvnVBEM('-distributions', copy(chosenPrior, K, 1), '-mixingPrior', mixPrior);
fitted{4} = fit(model{4}, galaxies, '-verbose', true, '-maxIter', 500, '-tol', 1e-10);
marginalDist{4} = marginalizeOutParams(fitted{4});

for m=1:numel(method)
  for k=1:K
    if(m == 1 || m == 2)
      meanEst(k,m) = mean(fitted{m}.samples.mu{k});
    elseif(m == 3)
      meanEst(k,m) = mean(fitted{m}.distributions{k});
    else
      meanEst(k,m) = mean(marginalDist{4}{k});
    end
  end
end
meanEst = sort(meanEst);
s{1,:} = sprintf('%s \t %s \t %s \t %s \t\t %s \n', 'Cluster', method{:});
for k=1:K
  s{k+1,:} = sprintf('%d \t\t %3.2f \t\t %3.2f \t\t\t %3.2f \t\t %3.2f \n', k, meanEst(k,:));
end
disp(catString(s,''))

disp(sprintf('Gibbs methods based on model averaging (%d samples)\n', Nsamples - Nburnin));
disp('(VB) EM based on posterior parameters with a Normal Inverse Wishart Prior')

% Visualizations
figure();
for m=1:numel(method)
  subplot(2,2,m); hold on;
  plot(fitted{m},'plotArgs', {'linewidth', 3});
  line([galaxies';galaxies'], [zeros(1,n);exp(logprob(fitted{m}, galaxies))'], 'color', 'red');
  title(method{m});
end
suptitle(sprintf('Fitting the Galaxy Dataset to %d mixtures \n (red lines indicate the location of the actual (scaled) data)', K));
printTitle = 'galaxyGibbsVsEMVsVB';
if printPmtk, printPmtkFigures(printTitle); end;