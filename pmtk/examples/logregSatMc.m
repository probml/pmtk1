%% Binary Classification of SAT Data via Bayesian Logistic Regression
% We use various methods to get posterior over w
% - MCMC
% - Importance sampling
% - Laplace

setSeed(0);
stat = load('satData.txt');

y = stat(:,1);                      % class labels
X = stat(:,4);                      % SAT scores
[X,perm] = sort(X,'ascend');        % sort for plotting purposes
y = y(perm); % 0,1
D = DataTable(X, y);

%T = ChainTransformer({StandardizeTransformer(false)});
%m = LogregBinaryMc('-transformer', T, '-lambda', 1e-3);
lambda = 1e-3;
models = { LogregBinaryMc('-lambda', lambda, '-fitEng', LogregBinaryImptceSampleFitEng()), ...
  LogregBinaryMc('-lambda', lambda, '-fitEng', LogregBinaryMhFitEng()), ...
  LogregBinaryLaplace('-lambda', lambda, '-predMethod', 'mc')};

for mi=1:length(models)
  m = models{mi};
  m = fit(m,D);
  L = sum(logprob(m,D))
  if mi<=2, L2 = sum(logprob(m,D,2)), end % debugging
  
  pw = getParamPost(m);
  figure; 
  for i=1:2
    pwi = marginal(pw, i);
    subplot(2,2,i); plot(pwi); title(sprintf('w%d',i));
  end
  
  [yhat, pred] = predict(m,D);
  [Q5,Q95] = credibleInterval(pred);
  med = median(pred);
  subplot(2,2,3); hold on
  plot(X, y, 'ko', 'linewidth', 3, 'markersize', 12);
  for i=1:length(y)
    line([X(i) X(i)], [Q5(i) Q95(i)],   'linewidth', 3);
    plot(X(i), med(i), 'rx', 'linewidth', 3, 'markersize', 12);
  end
end

