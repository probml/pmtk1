%% Plotting Binomial Distributions
%#testPMTK
muValues = [1/2 1/4 3/4 0.9];
N = 10;
figure;
for i=1:4
    subplot(2,2,i)
    b = BinomDist('-N',N,'-mu',muValues(i));
    plot(b);
    title(sprintf('mu=%5.3f', muValues(i)))
end
if doPrintPmtk, doPrintPmtkFigures('binomDistPlot'); end;
