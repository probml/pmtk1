
clear all

makePolyData;
doPrint = 0;

figure(1);clf
folder = 'C:\kmurphy\PML\Figures';


xtrain1 = rescaleData(xtrain);
xtest1 = rescaleData(xtest);

deg = 14;
Xtrain = degexpand(xtrain1, deg, 1);
Xtest = degexpand(xtest1, deg, 1);

lambdas = logspace(-10,1,10)
%lambdas = [100 10 5 1 0.1 0.01 0.001 0.0001 0];
for k=1:length(lambdas)
  lambda = lambdas(k);
  w = ridgeQR(Xtrain, ytrain, [], [], lambda, 0);

 ypredTest = Xtest*w;
 ypredTrain = Xtrain*w;
 testMse(k) = mean((ypredTest - ytest).^2);
 trainMse(k) = mean((ypredTrain - ytrain).^2);
 scatter(xtrain,ytrain,'b','filled');
 hold on;
 plot(xtest, ypredTest, 'k', 'linewidth', 3);
 hold off
 title(sprintf('lambda %5.3f, test mse %5.3f', lambda, testMse(k)))
 set(gca,'ylim',[-10 15]);
 set(gca,'xlim',[-1 21]);
 if doPrint
   fname = sprintf('%s/polyfitRidgeCV%5.3f.eps', folder, lambda);
   print(gcf, '-depsc', fname);
 end
 %pause
end

% CV
nfolds = -1;
Ntrain = size(Xtrain,1);
[trainfolds, testfolds] = Kfold(Ntrain, nfolds);
errors = zeros(length(lambdas), Ntrain); % errors(k,i)
for k=1:length(lambdas)
  lambda = lambdas(k);
  for i=1:length(trainfolds)
    XtrainFold = Xtrain(trainfolds{i},:);
    ytrainFold = ytrain(trainfolds{i});
    XtestFold = Xtrain(testfolds{i},:);
    ytestFold = ytrain(testfolds{i});
    w = ridgeQR(XtrainFold, ytrainFold, [], [], lambda, 0);
    ypredTest = XtestFold*w;
    err = (ypredTest - ytestFold).^2;
    errors(k, testfolds{i}) = err;
  end
end
errorRate = mean(errors,2);
errorRateSE = std(errors, 0, 2) / sqrt(Ntrain);

figure(3);clf
hold on
%ndx = log(lambdas);
ndx = dofRidge(Xtrain, lambdas);
plot(ndx, errorRate, 'ko-', 'linewidth', 2, 'markersize', 12);
plot(ndx, trainMse, 'bs:', 'linewidth', 2, 'markersize', 12);
plot(ndx, testMse, 'rx-', 'linewidth', 2, 'markersize', 12);
errorbar(ndx, errorRate, errorRateSE, 'k');
legend('CV', 'train', 'test')
xlabel('dof')
ylabel('mse')
set(gca,'ylim',[0 50])

if doPrint
  fname = sprintf('%s/polyfitDemoUcurve.eps', folder)
  print(gcf, '-depsc', fname);
end

keyboard

if 0
lambdas = logspace(-10,1,10)
for i=1:length(lambdas)
  [w, mseTrain(i), mseTest(i)] = ridgeFit(Xtrain, ytrain, Xtest, ytrueTest, lambdas(i));
end
  figure(2);clf
semilogx(lambdas, mseTrain, 'ro-', 'linewidth', 2, 'markersize', 12);
hold on
semilogx(lambdas, mseTest, 'kx:', 'linewidth', 2, 'markersize', 12);
xlabel('log10(\lambda)')
ylabel('MSE')
legend('train','test')
title(sprintf('mean squared error vs log regularizer, for polynomial of degree %d', deg))
end
