%% Poisson Plot Demo
%#testPMTK
lambdas = [0.1 1 10 20];
figure;
for i=1:4
    subplot(2,2,i)
    plot(PoissonDist(lambdas(i)));
    title(sprintf('Poi(%s=%5.3f)', '\lambda', lambdas(i)))
end