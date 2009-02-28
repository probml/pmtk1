%% Gibbs Sampling from the prior of the undirected water sprinkler
%#testPMTK
setSeed(0);
dgm = mkSprinklerDgm();
ugm = convertToUgm(dgm);

% Exact
ugmExact = ugm;
ugmExact.infMethod = 'enum'; 
Texact = zeros(2,4);
marg = marginal(ugmExact, {1,2,3,4,[1,2,3,4]});

for j=1:4
  %Texact(:,j) = pmf(marginal(ugmExact, j));
end
jointExact = pmf(marginal(ugmExact, [1 2 3 4]));
%joint = convertToTabularFactor(ugm);
%assert(approxeq(joint.T, jointExact))
disp('exact marginals')
Texact %#ok
figure; bar(jointExact(:)); title('exact joint')

break

% Gibbs
N = 100;
ugmGibbs = ugm;
ugmGibbs.infEng =  GibbsInfEng('Nsamples', N, 'verbose', true, 'Nchains', 2);
ugmGibbs = condition(ugmGibbs);
Tgibbs = zeros(2,4);
for j=1:4
  Tgibbs(:,j) = pmf(marginal(ugmGibbs, j));
end
jointGibbs = pmf(marginal(ugmGibbs, [1 2 3 4]));
disp('gibbs marginals')
Tgibbs %#ok
figure; bar(jointGibbs(:)); title('gibbs joint')

disp('convergence diagnostics')
ugmGibbs.infEng.convDiag
