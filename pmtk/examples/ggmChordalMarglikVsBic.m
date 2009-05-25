%% Compare marginal likelihood and BIC score on all decomposable GGMs

function ggmChordalMarglikVsBic()

folder = []; %'C:\kmurphy\Students\Baback\qualityOfBic';

d = 4; graphType = 'chain';
ns = [10 20 50];
priors = {'eb', 'vague'};
firstFig = true;
for ni=1:length(ns)
  n = ns(ni);
  seed = 0;
  setSeed(seed);
  Gtrue = UndirectedGraph('type', graphType, 'nnodes', d);
  truth = UgmGaussDist(Gtrue, [], []);
  truth = mkRndParams(truth);
  Y = sample(truth, n); 
  v = var(Y(:))
  Y = standardize(Y);
  %v = var(Y(:))
   
  for pi=1:length(priors)
    prior = priors{pi};
    switch prior
      case 'eb', Phi = var(Y(:))*eye(d); delta = 3;
      case 'vague', Phi = 1*eye(d); delta = 3;
    end
    GGMs = UgmGaussChordalDist.mkAllGgmDecomposable(delta, Phi);
    NG = length(GGMs);
    logpostG = zeros(1,NG);
    for i=1:NG
      logpostG(i) =  logmarglik(GGMs{i}, 'data', Y);
    end
    postG{pi} = exp(normalizeLogspace(logpostG));
    
    figure;
    bar(postG{pi});
    ttl = sprintf('p(G|D), prior %s, d=%d, n=%d', prior, d, n);
    title(ttl);
    if firstFig
      set(gca, 'ylim', [0 max(postG{pi})*1.01]);
      ylim = get(gca, 'ylim'); 
      firstFig = false;
    else
      set(gca, 'ylim', ylim);
    end
    %if ~isempty(folder), printPmtkFigures(safeStr(ttl), folder); end
    if ~isempty(folder), print(gcf, '-dpng', fullfile(folder,safeStr(ttl))); end
  end
  
  ll = zeros(1,NG); BIC = cell(1,2);
  for i=1:NG
    tmp = fit(UgmGaussDist(GGMs{i}.G), Y, '-shrink', true);
    ll(i) = sum(logprob(tmp, Y),1);
    df = dof(tmp);
    BIC{1}(i) = ll(i) - df*log(n)/2;
    BIC{2}(i) = ll(i)  - df*log(n)/2 - df/2*log(2*pi);
  end
  BICnames = {'BIC', 'fancyBIC'};
  for bi=1:2
    figure;
    BIC{bi} = exp(normalizeLogspace(BIC{bi}));
    bar(BIC{bi});
    set(gca, 'ylim', ylim);
    ttl = sprintf('%s, d=%d, n=%d', BICnames{bi}, d, n);
    title(ttl);
    if ~isempty(folder), print(gcf, '-dpng', fullfile(folder,safeStr(ttl))); end
    %if ~isempty(folder), printPmtkFigures(safeStr(ttl), folder); end
  end
  
  drawnow
  restoreSeed;
end % for ni
end


