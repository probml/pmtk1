function rainfallBootstrapDemo()

% Bootstrap estimate of sampling distribution of MOM and MLE for Gamma distribution
% MOM = method of moments, MLE = maximum likelihood estimate
% Based on Rice (1995) p251, 258
% We see that the sampling distribution of the MLE is much narrower than that of the MOM

X = dlmread('rainfallData.txt');
X = X'; X = X(:); % concatenate across rows, not columns
X = X(1:end-5); % removing trailing 0s

B  = 1000;
[asampMOM, bsampMOM] = bootstrap(B, X, @gamMOM);
[asampMLE, bsampMLE] = bootstrap(B, X, @gamMLE);

figure;
subplot(2,2,1)
hist(asampMOM);
title(sprintf('a MOM, se = %5.3f', std(asampMOM)));
set(gca,'xlim',[0.1 0.7])

subplot(2,2,2)
hist(bsampMOM);
title(sprintf('b MOM, se = %5.3f', std(bsampMOM)))
set(gca,'xlim',[0.5 3.5])

subplot(2,2,3)
hist(asampMLE); 
title(sprintf('a MLE, se = %5.3f', std(asampMLE)))
set(gca,'xlim',[0.1 0.7])

subplot(2,2,4)
hist(bsampMLE);
title(sprintf('b MLE, se = %5.3f', std(bsampMLE)))
set(gca,'xlim',[0.5 3.5])

if doPrintPmtk, printPmtkFigures('rainfallBootstrapDemo'); end;
%%%%%%
function [asamp, bsamp] = bootstrap(B, X, estimator)

N = length(X);
[agen, bgen] = feval(estimator, X);
Xsamp = gamrnd(agen, 1/bgen, B, N);
for b=1:B
  [asamp(b), bsamp(b)] = feval(estimator, Xsamp(b,:));
end

