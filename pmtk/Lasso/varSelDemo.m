function varSelDemo()

n = 100;
ds = 10:50:500;
sigma = 0.1;
sparsity = 0.1;
lassoSignConsistent = false; % faster to generate data if not require sign consistency
ntrials = 2;
nexpts = length(ds);
L2 = zeros(ntrials, nexpts);
precision = zeros(ntrials, nexpts);
recall = zeros(ntrials, nexpts);
for expt=1:nexpts
  fprintf('expt %d of %d\n', expt, nexpts);
  d = ds(expt);
  r = floor(sparsity*d);
  [L2(:,expt), precision(:,expt), recall(:,expt)] = ...
        helper(n,d,r,lassoSignConsistent,sigma,ntrials);
end

str = sprintf('n %d, sigma %3.2f, sparsity %3.2f, SC %d', n, sigma, sparsity, lassoSignConsistent);

figure; errorbar(ds, mean(L2), std(L2)/sqrt(ntrials));
title(sprintf('NMSE %s', str))
xlabel('d')

figure; errorbar(ds, mean(precision), std(precision)/sqrt(ntrials));
title(sprintf('precision, %s', str))
xlabel('d')

figure; errorbar(ds, mean(recall), std(recall)/sqrt(ntrials));
title(sprintf('recall, %s', str))
xlabel('d')

keyboard


%{

% r = number of relevant variables (out of r)
n = 50; 
ds = [n/2 2*n];
sigmas = [0.1];
signCons = [false true]; % sign consistency
ntrials = 5;
nexpts = length(ds)* length(sigmas)* length(signCons)
expt = 1;
for di=1:length(ds)
  for si=1:length(sigmas)
    for ci=1:length(signCons)
      fprintf('expt %d of %d\n', expt,nexpts);
      d = ds(di); sigma = sigmas(si); r = ceil(0.1*d);
      lassoSignConsistent = signCons(ci);
      [nmse(expt,:), precision(expt,:), recall(expt,:)] = ...
        helper(n,d,r,lassoSignConsistent,sigma,ntrials);
      name{expt} = sprintf('n%d,d%d,r%d,s%3.2f,c%d', ...
        n,d,r,sigma, lassoSignConsistent);
      expt = expt + 1;
    end
  end
end
figure;boxplot(nmse','labels',name); title('L2 loss');
figure;boxplot(precision','labels',name); title('precision');
figure;boxplot(recall','labels',name); title('recall');
keyboard
%}

end

function [nmse, precision, recall] = helper(n,d,r,lassoSignConsistent,sigma,ntrials)
nmse = zeros(1,ntrials); precision = zeros(1, ntrials); recall = zeros(1,ntrials);
for t=1:ntrials
  setSeed(t);
  [X,y,Wtrue] = bolassoMakeData(n,d,r,1,lassoSignConsistent,sigma);
  trueSupport = find(Wtrue ~= 0); % 1 to r
  [estSupport,West] = larsSelectSubsetCV(X,y);
  
  %{
  % to use vanilla lasso, set nbootstraps = 0
  % we select the optimal subset s from amongst those on the lars path
  % using CVerror(what(s)), where what(s) = X(:,s)\y is the OLS.
  tic
  [estSupport2,West2] = bolasso(X,y,'nbootstraps',0,...
    'statusBar', false, 'modelSelectionMethod','CV');
  toc
  assert(isequal(estSupport, estSupport2))
  %}
  
  ncorrect = length(intersect(estSupport, trueSupport));
  ncalled = length(estSupport);
  ntrue = length(trueSupport);
  precision(t) = ncorrect/ncalled;
  recall(t) = ncorrect/ntrue;
  Wtrue = [0; Wtrue]; % truth is zero mean
  nmse(t) = norm(West-Wtrue)/norm(Wtrue);
end
end
