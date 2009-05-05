%% Sample Dist Example
%#testPMTK
seed = 1;
setSeed(seed);
m = MvnDist();
m = mkRndParams(m, 2);
X = sample(m, 500);
mS = SampleBasedDist(X);
figure(1);clf
for i=1:2
    subplot2(2,2,i,1);
    mExact = marginal(m,i);
    mApprox = marginal(mS,i);
    [h, histArea] = plot(mApprox, 'useHisto', true);
    hold on
    [h, p] = plot(mExact, 'scaleFactor', histArea, 'plotArgs', {'linewidth', 2, 'color', 'r'});
    title(sprintf('exact mean=%5.3f, var=%5.3f', mean(mExact), var(mExact)));
    subplot2(2,2,i,2);
    plot(mApprox, 'useHisto', false);
    title(sprintf('approx mean=%5.3f, var=%5.3f', mean(mApprox), var(mApprox)));
end
figure(2);clf
plot(m, 'useContour', 'true');
hold on
plot(mS);
