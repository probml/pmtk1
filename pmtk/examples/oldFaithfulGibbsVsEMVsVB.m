%% Run full gibbs sampling, collapsed gibbs sampling, EM, and VB EM on the old Faithful dataset
% Print the resulting cluster means in a table
%#author Cody Severinski

cls; setSeed(0);
K = 2;
load oldFaith; [n,d] = size(X);
Nsamples = 1250; Nburnin = 250;

chosenPrior = MvnInvWishartDist('mu', mean(X)', 'Sigma', cov(X), 'dof', 3, 'k', 0.01);
mixPrior = DirichletDist(ones(K,1));

method = {'gibbs full', 'gibbs collapsed', 'EM', 'VB EM'};

model{1} = MixMvnGibbs('distributions',copy( MvnDist('-mu', zeros(d,1),'-Sigma', diag(ones(d,1)), '-prior', chosenPrior), K,1) ) ;
model{1}.mixingDistrib.prior = mixPrior;
fitted{1} = fit(model{1}, X, '-method', 'full', '-Nsamples', Nsamples, '-Nburnin', Nburnin, '-verbose', true);
fitted{2} = fit(model{1}, X, '-method', 'collapsed', '-Nsamples', Nsamples, '-Nburnin', Nburnin, '-verbose', true);

model{3} = MixMvn('-nmixtures', K, '-ndims', d) ;
model{3}.distributions = copy( MvnDist('-mu', zeros(d,1),'-Sigma', diag(ones(d,1)), '-prior', chosenPrior), K,1);
model{3}.mixingDistrib.prior = mixPrior;
fitted{3} = fit(model{3}, '-data', X);


model{4} = MixMvnVBEM('-distributions', copy(chosenPrior, K, 1), '-mixingPrior', mixPrior);
fitted{4} = fit(model{4}, X, '-verbose', true, '-maxIter', 500, '-tol', 1e-10);
marginalDist{4} = marginal(fitted{4});

for m=1:numel(method)
  for k=1:K
    if(m == 1 || m == 2)
      meanEst(k,[2*m-1, 2*m]) = mean(fitted{m}.samples.mu{k})';
    elseif(m == 3)
      meanEst(k,[2*m-1, 2*m]) = mean(fitted{m}.distributions{k})';
    else
      meanEst(k,[2*m-1, 2*m]) = mean(marginalDist{4}{k});
    end
  end
end
meanEst = sort(meanEst);
s{1,:} = sprintf('%s \t %s \t %s \t %s \t\t %s \n', 'Cluster', method{:});
for k=1:K
  s{k+1,:} = sprintf('%d \t\t [%3.2f, %3.2f] \t [%3.2f, %3.2f] \t\t [%3.2f, %3.2f] \t [%3.2f, %3.2f] \n', k, meanEst(k,:));
end
% PMTK1
disp(cellString(s,''))
% PMTK2
%disp(catString(s,''))

disp(sprintf('Gibbs methods based on model averaging (%d samples)\n', Nsamples - Nburnin))
disp('(VB) EM based on posterior parameters with a Normal Inverse Wishart Prior')
