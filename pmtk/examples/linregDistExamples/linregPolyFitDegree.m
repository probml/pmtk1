%% Linear Regression with Polynomial Basis Expansions
% based on code code by Romain Thibaux
% (Lecture 2 from http://www.cs.berkeley.edu/~asimma/294-fall06/)
[xtrain, ytrain, xtest, ytestNoisefree, ytest] = polyDataMake('sampling','thibaux');
doPrint = 0;
figure(1);clf
scatter(xtrain,ytrain,'b','filled');
%title('true function and noisy observations')
folder = 'C:\kmurphy\PML\pdfFigures';
degs = 0:16;
for i=1:length(degs)
    deg = degs(i);
    m = linregDist;
    m.transformer =  chainTransformer({rescaleTransformer, polyBasisTransformer(deg)});
    m = fit(m, 'X', xtrain, 'y', ytrain);
    ypredTrain = mean(predict(m, xtrain));
    ypredTest = mean(predict(m, xtest));
    testMse(i) = mean((ypredTest - ytest).^2);
    trainMse(i) = mean((ypredTrain - ytrain).^2);
    testLogprob(i) = sum(logprob(m, xtest, ytest));
    trainLogprob(i) = sum(logprob(m, xtrain, ytrain));
    [CVmeanMse(i), CVstdErrMse(i)] = cvScore(m, xtrain, ytrain, ...
        'objective', 'squaredErr');
    [CVmeanLogprob(i), CVstdErrLogprob(i)] = cvScore(m, xtrain, ytrain, ...
        'objective', 'logprob');

    figure(1);clf
    scatter(xtrain,ytrain,'b','filled');
    hold on;
    plot(xtest, ypredTest, 'k', 'linewidth', 3);
    hold off
    title(sprintf('degree %d, train mse %5.3f, test mse %5.3f',...
        deg, trainMse(i), testMse(i)))
    set(gca,'ylim',[-10 15]);
    set(gca,'xlim',[-1 21]);
    if doPrint
        fname = sprintf('%s/polyfitDemo%d.pdf', folder, deg)
        pdfcrop; print(gcf, '-dpdf', fname);
    end
end


figure(3);clf
hold on
%plot(degs, -CVmeanMse, 'ko-', 'linewidth', 2, 'markersize', 12);
plot(degs, trainMse, 'bs:', 'linewidth', 2, 'markersize', 12);
plot(degs, testMse, 'rx-', 'linewidth', 2, 'markersize', 12);
%errorbar(degs, -CVmeanMse, CVstdErrMse, 'k');
xlabel('degree')
ylabel('mse')
legend('train', 'test')
if doPrint
    fname = sprintf('%s/polyfitDemoUcurve.pdf', folder)
    pdfcrop; print(gcf, '-dpdf', fname);
end
