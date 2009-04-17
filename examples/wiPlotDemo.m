%% Plot Some Wishart Distributions
%#testPMTK
seed = 2;
setSeed(seed);
S=randpd(2);
dofs = [20 3];
for i=1:length(dofs)
    dof = dofs(i);
    p=WishartDist(dof,S);
    M = mean(p);
    R = cov2cor(M);
    %h=figure;clf;
    plotSamples2d(p,9);
    suptitle(sprintf('Wi(dof=%3.1f, S), E=[%3.1f, %3.1f; %3.1f, %3.1f], %s=%3.1f', ...
        dof, M(1,1), M(1,2), M(2,1), M(2,2), '\rho', R(1,2)));
    %h=figure;clf
    plotMarginals(p);
end
