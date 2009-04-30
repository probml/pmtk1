%% Linear Regression with Polynomial Basis of different degrees
% based on code code by Romain Thibaux
% (Lecture 2 from http://www.cs.berkeley.edu/~asimma/294-fall06/)


[xtrain, ytrain, xtest, ytestNoisefree, ytest] = polyDataMake('sampling','thibaux');
Dtrain = DataTable(xtrain, ytrain);
Dtest = DataTable(xtest, ytest);

degs = [0 1 2 4 6 8 10 12 14];
Nm = length(degs);
models = cell(1,Nm); 
for m=1:length(degs)
  deg = degs(m);
  T = ChainTransformer({RescaleTransformer, PolyBasisTransformer(deg,false)});
  models{m} = Linreg('-transformer', T);
end
ML = ModelList(models);
ML = fit(ML, Dtrain);
mseTrain = zeros(1,Nm); mseTest = zeros(1,Nm);
for m=1:Nm
  figure;
  scatter(xtrain,ytrain,'b','filled');
  hold on;
  ypredTrain = predict(ML.models{m}, xtrain);
  mseTrain(m) = mse(ytrain,  ypredTrain); 
  ypredTest = predict(ML.models{m}, xtest);
  mseTest(m) = mse(ytest,  ypredTest);
  plot(xtest, ypredTest, 'k', 'linewidth', 3);
  hold off
  deg = degs(m);
  title(sprintf('degree %d', deg))
  %set(gca,'ylim',[-10 15]);
  %set(gca,'xlim',[-1 21]);
end

figure;
hold on
plot(degs, mseTrain, 'bs:', 'linewidth', 2, 'markersize', 12);
plot(degs, mseTest, 'rx-', 'linewidth', 2, 'markersize', 12);
xlabel('degree')
ylabel('mse')
legend('train', 'test')

placeFigures;


%% Model selection

% BIC-posterior
figure;
bar(degs, ML.posterior);
title('BIC posterior over models')
xlabel('polynomial degree')


% True posterior  
modelsConj = cell(1,Nm);
for m=1:length(degs)
  deg = degs(m);
  T = ChainTransformer({RescaleTransformer, PolyBasisTransformer(deg,false)});
  modelsConj{m} = LinregConjugate('-transformer', T);
end
% Use log marginal likelihood
MLconj = ModelList(modelsConj, '-selMethod', 'loglik');
MLconj = fit(MLconj, Dtrain);

figure;
bar(degs, MLconj.posterior);
title('posterior over models')
xlabel('degree')


