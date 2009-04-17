function NBbayesGaussDemo()
% Illustrate benefit of using T distribution instead of plugin
% for a naive Bayes classifier with 2 Gaussian features
% Written by Kevin Murphy and Hoyt Koepke

figure(3); clf
runDemo(0,1,1);

N = 200;
rand('seed', 0)
randn('seed', 0)
erdiff = zeros(N,2);
for i = 1:N
  erdiff(i,:) = runDemo(1,0,0);
end
figure(4);clf
%hist(erdiff(:,1) - erdiff(:,2), N/10)
hist(erdiff(:,1) - erdiff(:,2))
title(['Error rate of plugin - Bayes'])
fprintf('plugin %5.3f, bayes %5.3f\n', ...
	mean(erdiff(:,1)), mean(erdiff(:,2)))


%%%%%%%%%
function [errrate] = runDemo(withNoise, showplots, setrandseed)

symbols = {'r+', 'b*',  'gx', 'mx', 'r.', 'gs', 'c*'};
errrate = zeros(1,2);
K = 3;
if setrandseed
  rand('seed', 3)
  randn('seed', 3)
end

[Xtrain, Ytrain, Xtest, Ytest] = makeData(withNoise);

if showplots == 1
  subplot(2,2,1)
  for j = 1:K
    plot(Xtrain(1, Ytrain == j), Xtrain(2, Ytrain == j), symbols{j});
    hold on
  end
  title('training data')
  subplot(2,2,2)
  for j = 1:K
    plot(Xtest(1, Ytest == j), Xtest(2, Ytest == j), symbols{j});
    hold on
  end
  title('test data')
end

params = gaussianClassifierTrain(Xtrain', Ytrain);

params.classPrior = normalize(ones(1,K)); % force it to be uniform
[post, Ypred{1}] = gaussianClassifierApply(Xtest', params, 1, 0);
[post, Ypred{2}] = gaussianClassifierApply(Xtest', params, 1, 1);
str{1} = 'plugin';
str{2} = 'bayes';

for i=1:2
  errors = (Ypred{i} ~= Ytest');
  nerr = sum(errors);
  errrate(i) = mean(errors);
  if showplots == 1
    subplot(2,2,2+i)
    for j = 1:K
      plot(Xtest(1, Ypred{i} == j), Xtest(2, Ypred{i} == j), symbols{j});
      hold on
    end
    plot(Xtest(1, errors), Xtest(2, errors), 'ko');
    %title(sprintf('%s %d errors %5.3f', str{i}, nerr, errrate(i)));
    title(sprintf('%s %d errors', str{i}, nerr))
  end
end

  
%%%%%%%%%%

function [X, Y] = makeDataHelper(nPts_n)

% Generate three gaussian clusters and one with uniform
% contaimination, both in 2d

mu = [0, 0.6;
      -0.3,-0.4;
      1,-0.5]';

var = [0.6,0.4,0.6];

N = sum(nPts_n);
dim = size(mu,1);
X = zeros(2, N);
Y = zeros(1, N);

curidx = 1;
for j = 1:3
  X(:,curidx:curidx+nPts_n(j)-1) = repmat(mu(:,j), 1, nPts_n(j))+var(j).*randn(dim, nPts_n(j));
  Y(curidx:curidx+nPts_n(j)-1) = j;
  curidx = curidx + nPts_n(j);
end


%%%%%%%%%

function [Xtrain,Ytrain, Xtest, Ytest] = makeData(noise_okay)

[Xtrain, Ytrain] = makeDataHelper([2 5 8]*1);
[Xtest, Ytest] = makeDataHelper([2 5 8]*20);

%%%%%%%%%%

function [Xtrain,Ytrain, Xtest, Ytest] = makeDataOld(noise_okay)

% Generate three gaussian clusters and one with uniform
% contaimination, both in 2d

Ntrain = 20;
Ntest = 200;

mu = [0, 0.6;
      -0.3,-0.4;
      1,-0.5]';

var = [0.6,0.4,0.6];
%nPts_n = [80,150,105]*3;
nPts_n = [5,10,20]*10;
nPts_u = [30,0,15]*10*(noise_okay == 1);

N = sum(nPts_n) + sum(nPts_u);
K = length(nPts_n);
dim = size(mu,1);

X = zeros(2, N);
Y = zeros(1, N);


curidx = 1;

for j = 1:K
  X(:,curidx:curidx+nPts_n(j)-1) = repmat(mu(:,j), 1, nPts_n(j))+var(j).*randn(dim, nPts_n(j));
  Y(curidx:curidx+nPts_n(j)-1) = j;
  curidx = curidx + nPts_n(j);
  X(:,curidx:curidx+nPts_u(j)-1) = (2-4*rand(dim, nPts_u(j)));
  Y(curidx:curidx+nPts_u(j)-1) = j;
  curidx = curidx + nPts_u(j);
end

idx = randperm(N);
Xtrain = X(:,idx(1:Ntrain));
Ytrain = Y(idx(1:Ntrain));
Xtest = X(:,idx(Ntrain+1:Ntrain+Ntest));
Ytest = Y(idx(Ntrain+1:Ntrain+Ntest));

%%%%%%%%%
