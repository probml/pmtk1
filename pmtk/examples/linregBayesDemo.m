%% Bayesian Linear Regression with Various Bases and sample sizes
function linregBayesDemo()

setSeed(0);
basis = {'quad', 'rbf'};
sampling = {'sparse', 'dense'};
deg = 2;
for j=1:length(basis)
  for k=1:length(sampling)
    helperBasis(...
      'deg', deg, 'basis', basis{j}, 'sampling', sampling{k}, ...
      'plotErrorBars', true);
  end
end

end


function helperBasis(varargin)
[sampling, deg, basis, plotErrorBars, plotBasis, plotSamples] = ...
  process_options(varargin, ...
  'sampling', 'sparse', 'deg', 3, 'basis', 'rbf', ...
  'plotErrorBars', true, 'plotBasis', true, ...
  'plotSamples', true);

[xtrain, ytrain, xtest, ytestNoisefree, ytestNoisy, sigma2] = polyDataMake(...
  'sampling', sampling, 'deg', deg);
switch basis
  case 'quad'
    T = PolyBasisTransformer(2, false);
  case 'rbf'
    T = RbfBasisTransformer(10, 1);
end
m = LinregConjugate('-lambda', 0.001, '-transformer', T, '-sigma2', sigma2);
Dtrain = DataTable(xtrain, ytrain);
m = fit(m, Dtrain);
[ypredTest, ypredTestProb] =  predict(m,xtest);

figure; hold on;
h = plot(xtest, ypredTest,  'k-');
set(h, 'linewidth', 3)
h = plot(xtest, ytestNoisefree,'b-');
set(h, 'linewidth', 3)
%h = plot(xtrain, mean(ypredTrain), 'gx');
%set(h, 'linewidth', 3, 'markersize', 12)
h = plot(xtrain, ytrain, 'ro');
set(h, 'linewidth', 3, 'markersize', 12)
grid off
legend('prediction', 'truth', 'training data', ...
  'location', 'northwest')
if plotErrorBars
  NN = length(xtest);
  ndx = 1:3:NN;
  sigma = sqrt(var(ypredTestProb));
  mu = mean(ypredTestProb);
  h=errorbar(xtest(ndx), mu(ndx), 2*sigma(ndx));
  set(h, 'color', 'k');
end
title(sprintf('truth = degree %d, basis = %s, n=%d', ...
  deg, basis, length(xtrain)));

% superimpose basis fns
if strcmp(basis, 'rbf') && plotBasis
  ax = axis;
  ymin = ax(3);
  h=0.1*(ax(4)-ax(3));
  [Ktrain,T] = train(T, xtrain);
  Ktest = test(T, xtest);
  K = size(Ktest,2); % num centers
  for j=1:K
    plot(xtest, ymin + h*Ktest(:,j));
  end
end

% Plot samples from the posterior
if ~plotSamples, return; end
figure;clf;hold on
plot(xtest,ytestNoisefree,'b-','linewidth',3);
h = plot(xtrain, ytrain, 'ro');
set(h, 'linewidth', 3, 'markersize', 12)
grid off
nsamples = 10;
for s=1:nsamples
  ws = sample(m.wDist);
  w0 = ws(1);  w = ws(2:end);
  ms = Linreg('-w', w(:), '-w0', w0, '-sigma2', sigma2, '-transformer', T);
  %keyboard
  [xtrainT, ms.transformer] = train(ms.transformer, xtrain);
  ypred = predict(ms, xtest);
  plot(xtest, ypred, 'k:', 'linewidth', 1);
end
title(sprintf('truth = degree %d, basis = %s, n=%d', deg, basis, length(xtrain)))
end
