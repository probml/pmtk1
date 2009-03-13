%% Gibbs Sampling from the prior of the undirected water sprinkler
%#testPMTK
setSeed(0);
dgm = mkSprinklerDgm();
ugm = convertToUgm(dgm);

% Exact
ugmExact = ugm;
ugmExact.infMethod = EnumInfEng(); 
Jexact = pmf(marginal(ugmExact, [1 2 3 4]));
figure; bar(Jexact(:)); title('exact joint')


% Gibbs
ugmGibbs = ugm;
ugmGibbs.infMethod = GibbsInfEng('Nsamples', 100, 'verbose', true, 'Nchains', 2);
[jointGibbs, junk, convDiag] = marginal(ugmGibbs, [1 2 3 4]);
disp('gibbs marginals')
Jgibbs = pmf(jointGibbs);
figure; bar(Jgibbs(:)); title('gibbs joint')

disp('convergence diagnostics')
convDiag %#ok
