clear all
%load facesOlivetti_trainTest.mat
% h = 112; w = 92;

doPrint = 0;
folder = 'C:\kmurphy\PML\Figures';

load('olivettifaces.mat'); % 0 to 255, from http://www.cs.toronto.edu/~roweis/data.html
X=faces'; clear faces; % 4096x400  (64x64=4096) 
% 10 images per person, 40 people
y = repmat((1:40),10,1); y = y(:);
[n d] = size(X);
h = 64; w = 64;
seed = 0; rand('state', seed); randn('state', seed);

trainndx = []; testndx = [];
for i=1:40
  trainndx = [trainndx (1:8)+10*(i-1)];
  testndx = [testndx (9:10)+10*(i-1)];
end

Xtrain = X(trainndx,:);
Xtest = X(testndx,:);
ytrain = y(trainndx);
ytest = y(testndx);

if 1
  % single test image per person to make prettier picture
  Xtest = Xtest(1:2:end, :); 
  ytest = ytest(1:2:end);
  %Xtest = Xtest(2:2:end, :); 
  %ytest = ytest(2:2:end);
end

if 0
   % half the training set for speed
Xtrain = Xtrain(1:2:end, :);
ytrain = ytrain(1:2:end);
end

Ntest= length(ytest);


% Show test images
figure(1); clf
plotFaces(Xtest, 1:Ntest, ytest, ytest)
title('test images')
drawnow
fname = sprintf('%s/faceRecTest.eps', folder)
if doPrint, print(gcf, '-depsc', fname), end

rank(Xtrain)
[B, Z, evals, Xrecon, mu] = pcaMLABA(Xtrain);

[Ntrain d] = size(Xtrain);
Ntest = size(Xtest,1);
maxK = Ntrain;
Ks = [1 2 4 6 8 10 15 20 25 50 100 120 140 rank(Xtrain)];
clear err


for k=1:length(Ks)
  K = Ks(k)
  XtrainProj = ?
  XtrainRecon = ?
  mseTrain(k) = mean((XtrainRecon(:)-Xtrain(:)).^2);
  XtestProj = ?
  XtestRecon = ?
  mseTest(k) = ?

  dst{K} = sqdist(XtestProj', XtrainProj'); % dst(test, train);
  closest = ?
  yhat = ytrain(closest); 
  err(k) = sum(yhat ~= ytest)
end

figure(2);clf;
plot(Ks,err,'-o')
xlabel('PCA dimensionality')
ylabel('number of misclassification errors on test set')


% Now do matching in image space (no PCA)
dstRaw = sqdist(Xtest', Xtrain');
[junk, closest] = min(dstRaw,[],2);
yhat = ytrain(closest); 
errDirect = sum(yhat ~= ytest)


% Now show closest match
figure(3);clf
K = 4;
[junk, closest] = min(dst{K},[],2);
ytestPred = ytrain(closest);
plotFaces(Xtrain, closest,  ytest, ytestPred);
title(sprintf('closest match in training set using K=%d',K))

figure(4);clf
K = 25;
[junk, closest] = min(dst{K},[],2);
ytestPred = ytrain(closest);
plotFaces(Xtrain, closest,  ytest, ytestPred);
title(sprintf('closest match in training set using K=%d',K))
