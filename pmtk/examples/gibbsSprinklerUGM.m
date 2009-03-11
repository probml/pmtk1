%% Gibbs Sampling from the prior of the undirected water sprinkler
%#testPMTK
setSeed(0);
dgm = mkSprinklerDgm();
ugm = convertToUgmTabular(dgm);

% Exact
ugmExact = ugm;
ugmExact.infEng = EnumInfEng(); 
Jexact = pmf(marginal(ugmExact, [1 2 3 4]));
figure; bar(Jexact(:)); title('exact joint')


% Gibbs
ugmGibbs = ugm;
ugmGibbs.infEng = GibbsInfEng('Nsamples', 100, 'verbose', true, 'Nchains', 2);
[jointGibbs, junk, convDiag] = marginal(ugmGibbs, [1 2 3 4]);
disp('gibbs marginals')
Jgibbs = pmf(jointGibbs);
figure; bar(Jgibbs(:)); title('gibbs joint')

disp('convergence diagnostics')
convDiag %#ok
