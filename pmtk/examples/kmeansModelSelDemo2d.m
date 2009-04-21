%% Select K for K-means and GMM using a test set
% We see that reconstruction error on test set goes down with increasing K
% However loglikelihood for GMM has U shape curve
% Thus we cannot use cross-validation for K-means, but can for GMM
%#broken
%clear all
setSeed(0);
d = 2; K = 3;
Ntrain = 100; Ntest = 100;

M = MixMvn(K, d);
M.distributions{1}.mu = [1 1]';
M.distributions{2}.mu = -[1 1]';
M.distributions{3}.mu = [0 0]';
for k=1:K
  M.distributions{k}.Sigma = 0.1*eye(d);
end
Xtrain = sample(M, Ntrain);
Xtest = sample(M, Ntest);

%{ 
%netlab version
mix = gmm(d, ncentres, 'spherical');
mix.centres = [1 1; -1 -1; 0 0]; % mu(k,:)
mix.covars = [0.1 0.1 0.1];
Xtrain = gmmsamp(mix, Ntrain);
Xtest = gmmsamp(mix, Ntrain);
%}

%Ks = [1 2 3 4 5  10 15 20 25];
Ks = [2 5  10 20  30];
%Ks = [2 10 30];
[nr,nc] = nsubplots(length(Ks));


figure;
for i=1:length(Ks)
  K = Ks(i);
  setSeed(0);
  mu = kmeansSimple(Xtrain, K);
  Xhat = kmeansDecode(kmeansEncode(Xtest, mu), mu);
  
  %{
  % transformer version - calls same underlying code
  setSeed(0);
  [Ztrain, T] = train(KmeansTransformer(K), Xtrain);
  [Ztest, Xhat2] = test(T, Xtest);
  assert(approxeq(Xhat, Xhat2))
  %}
  
  mse(i) = mean(sum((Xhat - Xtest).^2,2)); %#ok
  
  subplot(nr,nc,i)
  plot(Xtrain(:,1), Xtrain(:,2), '.');
  hold on
  for k=1:K
    plot(mu(k,1), mu(k,2), 'rs', 'linewidth', 3);
  end
  %title('train')
  %plot(Xtest(:,1), Xtest(:,2), 'ko'); hold on;
  %plot(Xhat{i}(:,1), Xhat{i}(:,2), 'rx');
  title(sprintf('K=%d, mse=%5.4f', K, mse(i)))
end
figure; plot(Ks, mse, 'o-')
title('MSE on test set vs K')


figure;
[nr,nc] = nsubplots(length(Ks));
for i=1:length(Ks)
  K = Ks(i);
  
  M = MixMvn(K, d);
  M.mixingDistrib.prior = 'none';
  for k=1:K
    M.distributions{k}.prior = 'none'; % NIW prior means nonmonotonic.
  end
  M.fitEng.verbose = true;
  %M.fitEng.maxIter= 8; % otherwise becomes singular...
  %M.fitEng.nrestarts = 1;
  
  M = fit(M, Xtrain);
  nll(i) = -sum(logprob(M, Xtest));
  mu = zeros(K,d);
  for k=1:K
    mu(k,:) = M.distributions{k}.mu';
  end
  
  %{
  %netlab version
  options = foptions;
  mix = gmm(d, K, 'spherical');
  mix = gmmem(mix, Xtrain, options);
  nll(i) = -sum(log(gmmprob(mix, Xtest)));
  mu = mix.centres;
  %}
  
  subplot(nr,nc,i)
  plot(Xtrain(:,1), Xtrain(:,2), '.');
  hold on
  for k=1:K
    plot(mu(k,1), mu(k,2), 'rs', 'linewidth', 3);
  end
  %title('train')
  %plot(Xtest(:,1), Xtest(:,2), 'ko'); hold on;
  %plot(Xhat{i}(:,1), Xhat{i}(:,2), 'rx');
  title(sprintf('K=%d, nll=%5.4f', K, nll(i)))
end

figure; plot(Ks, nll, 'o-')
title('NLL on test set vs K')
