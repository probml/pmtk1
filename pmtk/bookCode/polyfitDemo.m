% based on code code by Romain Thibaux
% (Lecture 2 from http://www.cs.berkeley.edu/~asimma/294-fall06/)

clear all

makePolyData;
doPrint = 0;

figure(1);clf
plot(xtrain,ytrueTrain,'k','linewidth',3);
hold on;
scatter(xtrain,ytrain,'r','filled');
hold off;
title('true function and noisy observations')

figure(2);clf
folder = 'C:\kmurphy\PML\Figures';

xtrain1 = rescaleData(xtrain);
xtest1 = rescaleData(xtest);
Xtrain = zeros(length(xtrain),0);
Xtest = zeros(length(xtest),0);
n = size(Xtrain,1);
degs = 0:n;
for i=1:length(degs)
  deg = degs(i);
 %sequentially add on higher powers
 Xtrain = [Xtrain, xtrain1.^deg];
 Xtest = [Xtest, xtest1.^deg];

 Xtrain2 = degexpand(xtrain1, deg, 1);
 assert(approxeq(Xtrain, Xtrain2))
 Xtest2 = degexpand(xtest1, deg, 1);
 assert(approxeq(Xtest, Xtest2))
 
 w = Xtrain \ ytrain; % least squares
 ypredTest = Xtest*w;
 ypredTrain = Xtrain*w;
 testMse(i) = mean((ypredTest - ytest).^2);
 trainMse(i) = mean((ypredTrain - ytrain).^2);
 scatter(xtrain,ytrain,'b','filled');
 hold on;
 plot(xtest, ypredTest, 'k', 'linewidth', 3);
 hold off
 title(sprintf('degree %d, test mse %5.3f', deg, testMse(i)))
 set(gca,'ylim',[-10 15]);
 set(gca,'xlim',[-1 21]);
 if doPrint
   fname = sprintf('%s/polyfitDemo%d.eps', folder, deg)
   print(gcf, '-depsc', fname);
 end
 %pause
 %if deg>=14, pause; end
end

% CV
nfolds = -1;
Ntrain = size(Xtrain,1);
[trainfolds, testfolds] = Kfold(Ntrain, nfolds);
errors = zeros(length(degs), Ntrain); % errors(k,i)
for k=1:length(degs)
  deg = degs(k);
  Xtrain = degexpand(xtrain1, deg, 1);
  for i=1:length(trainfolds)
    XtrainFold = Xtrain(trainfolds{i},:);
    ytrainFold = ytrain(trainfolds{i});
    XtestFold = Xtrain(testfolds{i},:);
    ytestFold = ytrain(testfolds{i});
    w = XtrainFold \ ytrainFold; % least squares
    ypredTest = XtestFold*w;
    %errorRateFold(k,i) = mean((ypredTest - ytestFold).^2);
    err = (ypredTest - ytestFold).^2;
    errors(k, testfolds{i}) = err;
  end
end
%errorRate = mean(errorRateFold, 2);
%errorRateSE = std(errorRateFold, 0, 2) / sqrt(nfolds);
errorRate = mean(errors,2);
errorRateSE = std(errors, 0, 2) / sqrt(Ntrain);

figure(3);clf
hold on
plot(degs, errorRate, 'ko-', 'linewidth', 2, 'markersize', 12);
plot(degs, trainMse, 'bs:', 'linewidth', 2, 'markersize', 12);
plot(degs, testMse, 'rx-', 'linewidth', 2, 'markersize', 12);
errorbar(degs, errorRate, errorRateSE, 'k');
legend('CV', 'train', 'test')
xlabel('degree')
ylabel('mse')
set(gca,'ylim',[0 50])

if doPrint
  fname = sprintf('%s/polyfitDemoUcurve.eps', folder)
  print(gcf, '-depsc', fname);
end
