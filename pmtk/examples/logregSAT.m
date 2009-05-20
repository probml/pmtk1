%% Binary Classification of SAT Data via Logistic Regression
% In this example, we classify whether or not a student will pass a
% course based on their SAT score, using logistic regression.
%#testPMTK
%% Load Data
setSeed(0);
stat = load('satData.txt');
%%
% The data is from Johnson and Albert p77 table 3.1
% Columns of stat:
% # pass [0 | 1]
% # all ones
% # all ones
% # SAT score
% # prerequisite grade where A=5,B=4,C=3,D=2,F=1
%%
% The data is in {0,1} 0 for fail, 1 for pass. 
y = stat(:,1);                      % class labels
X = stat(:,4);                      % SAT scores
[X,perm] = sort(X,'ascend');        % sort for plotting purposes
y = y(perm);
D = DataTable(X, y);
%% Fit via MLE
T = ChainTransformer({StandardizeTransformer(false)});
m = Logreg('-nclasses',2,'-transformer', T);
m = fit(m, D);  
%% Classify Training Examples
[yhat, pred] = predict(m,X);                      % predict on the training examples using MLE
yhat = mode(pred);                        % most probable class labels 
pmat = pmf(pred);                         % convert to a Matlab matrix
yprob = pmat(2,:)';                       % probability of passing given SAT score and fitted weights
%% Plot MLE
figure; hold on
plot(X, y , 'ko', 'linewidth', 3,   'markersize', 12);    
plot(X, yhat, 'b.',                 'markersize', 18);
plot(X, yprob,'xr', 'linewidth', 3, 'markersize', 12);
set(gca, 'ylim'    , [-0.2 1.6]                ,...
    'YTick'        , [0,0.5,1]                 ,...
    'YTickLabel'   , {'Fail  0','','Pass  1'}  ,...
    'FontSize'     , 12                        ,...
    'YGrid'        , 'on'                      ,...
    'box'          , 'on');
xlabel('SAT Score')
legend({'Actual','Predicted','p( passing | SAT Score , w )'},'Location','NorthWest');
title('MLE');
if doPrintPmtk, printPmtkFigures('logregGradeFitted'); end;
%%  Fit Using Laplace Approximation to the Posterior
% Here we fit in much the same way but use the Logreg_MvnDist and compute a
% Laplace approximation to the posterior. 
mBayes = LogregBinaryLaplace('-transformer',T, '-lambda', 1e-3, ...
  '-predMethod', 'mc', '-nsamples', 100);
mBayes = fit(mBayes,D);

pw = getParamPost(mBayes);
for i=1:2
  pwi = marginal(pw, i);
  figure; plot(pwi); title(sprintf('w%d',i));
end


%% Predict using Monte Carlo sampling of the Posterior Predictive
[yhat, pred] = predict(mBayes,X);
% Here we obtain error bars on our predictions by looking at the
% credible intervals.
[Q5,Q95] = credibleInterval(pred); 
med = median(pred);
figure; hold on
plot(X, y, 'ko', 'linewidth', 3, 'markersize', 12);
for i=1:length(y)
    line([X(i) X(i)], [Q5(i) Q95(i)],   'linewidth', 3);
    plot(X(i), med(i), 'rx', 'linewidth', 3, 'markersize', 12);
end
set(gca, 'ylim'    , [-0.1 1.6]                                    ,...
    'YTick'        , 0:0.25:1                                      ,...
    'YTickLabel'   , {'(Fail)  0','0.25','0.5','0.75','(Pass)  1'} ,...
    'FontSize'     , 12                                            ,...
    'YGrid'        , 'on'                                          ,...
    'box'          , 'on');
xlabel('SAT Score')
legend({'Actual','95% credible interval','p( passing | SAT Score , w )'},'Location','NorthWest');
title('Bayes');


%% Plot Posterior Predictive Samples
figure; hold on
plot(X, y, 'ko', 'linewidth', 3, 'markersize', 12);
for s=1:30
    plot(X, pred.samples(s,:), 'r-');
end
axis tight
set(gca, 'ylim'   , [-0.1 1.2]                                    ,...
    'YTick'       , 0:0.25:1                                      ,...
    'YTickLabel'  , {'(Fail)  0','0.25','0.5','0.75','(Pass)  1'} ,...
    'FontSize'    , 12                                            ,...
    'YGrid'       , 'on'                                          ,...
    'box'         , 'on');
xlabel('SAT Score');
title('Posterior Predictive Samples');

