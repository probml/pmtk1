%% Plot Histogram of Samples Generated From a DirichletDist
%#testPMTK
alphavec = [0.1, 1.0, 5.0]; 
alphavecstr = {'01', '1', '5'};
for j=1:length(alphavec)
  alpha = alphavec(j);
  setSeed(0);
  obj = DirichletDist(alpha*ones(1,5));
  n = 5;
  probs = sample(obj, n);
  figure;
  for i=1:n
      subplot(n,1,i); bar(probs(i,:))
      if i==1, title(sprintf('Samples from Dir %3.1f', alpha)); end
  end
  if doPrintPmtk, doPrintPmtkFigures(sprintf('dir%d', alphavecstr{j})); end;
  restoreSeed();
end