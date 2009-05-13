%% Compare various density estimators on various  data sets

for cts=0:1
  if cts
    pima = csvread('pimatr.csv',1,0);
    pima = pima(:,3:6); % 200x4
    datasets = { pima; };
    datasetNames = {'pima'};
    models = {...
      MvnDist('-covtype', 'full'), ...
      MvnDist('-covtype', 'diag'), ...
      MixMvnDist('-nmixtures', 5, '-covtype', 'full') ...
      };
    modelNames = {'full', 'diag', 'mixFull'};
  else
    foo = load('sachsDiscretized'); sachs = foo.X; % 5400x11
    foo = load('newsgroupUnique'); news = foo.X; % 10267x100
    datasets = {sachs}; % news
    datasetNames = 'sachs';
    models = {...
      DiscreteProdDist(), ...
      MixDiscrete('-nmixtures', 2, '-fitEng', EmMixModelEng('-nrestarts', 1));
      DgmTreeTabular() ...
      };
    modelNames = {'factored', 'mix2', 'tree'}
  end
   
pcMissing = 0.3;

for di=1:length(datasets)
 X = datasets{di};
 
  Nfolds = 2;
  N = size(X,1);
  setSeed(1);
  randomizeOrder = true;
  [trainfolds, testfolds] = Kfold(N, Nfolds, randomizeOrder);

  for m=1:length(models)
    for f=1:Nfolds
      Xtrain = X(trainfolds{f},:);
      Xtest = X(testfolds{f},:);
      M = fit(models{m}, Xtrain);
      ll = logprob(M, Xtest);
      NLL(f,m) = -sum(ll); %#ok
      
      [n, d] = size(Xtest);
      missing = rand(n,d) < pcMissing;
      Xmiss = Xtest;
      Xtest(missing) = NaN;
      XtestImpute = impute(M, Xtest);
      if cts
        imputeLoss(f,m) = sum(sum(XtestImpute - Xtest).^2)/numel(XtestImpute);
      else
        imputeLoss(f,m) = sum(sum(XtestImpute ~= Xtest))/numel(XtestImpute);
      end
    end
  end
  figure; boxplot(NLL, 'labels', modelNames);
  title(sprintf('NLL on %s', datasetNames{d}));
  drawnow
  
  figure; boxplot(imputeLoss, 'labels', modelNames);
  title(sprintf('imputation loss on %s', datasetNames{d}));
  drawnow
end % dataset

end % cts


