function digitsIIDclassify()

% naive Bayes classifier for digits
%Ntrain=5000, 83 errors in 500, rate = 0.166

load('mnistALL') % already randomly shuffled across classes
% train_images: [28x28x60000 uint8]
% test_images: [28x28x10000 uint8]
% train_labels: [60000x1 uint8]
% test_labels: [10000x1 uint8]

doPlot = 0;
if 1
  % to illustrate that shuffling the features does not affect classification performance
  perm  = randperm(28*28);
  mnist.train_images = reshape(mnist.train_images, [28*28 60000]);
  mnist.train_images = mnist.train_images(perm, :);
  mnist.train_images = reshape(mnist.train_images, [28 28 60000]);

  mnist.test_images = reshape(mnist.test_images, [28*28 10000]);
  mnist.test_images = mnist.test_images(perm, :);
  mnist.test_images = reshape(mnist.test_images, [28 28 10000]);
  doPlot = 0;
end

if 1
% test unpermuting
figure(1);clf;figure(2);clf; 
for i=1:9
  img =  mnist.test_images(:,:,i);
  y = mnist.test_labels(i);
  figure(1);
  subplot(3,3,i)
  imagesc(img);colormap(gray); axis off
  title(sprintf('true class = %d', y))
  
  img2(perm) = img(:);
  img2 = reshape(img2, [28 28]);
  figure(2);
  subplot(3,3,i)
  imagesc(img2); colormap(gray); axis off
  title(sprintf('true class = %d', y))
end
end


%trainSize = [10 50 100 250 500 1000 3000 5000];
%trainSize = [10 50 100 250 500];
trainSize = [500];
for trial=1:length(trainSize)
  Ntrain = trainSize(trial);

for digit=0:9
  modelNum = digit+1;
  [model(modelNum).pOn, model(modelNum).mu] = trainModel(mnist, digit, Ntrain);
end

nerr = 0;
Ntest = 500;
ndx = 1:Ntest;
Xtest = double(reshape(mnist.test_images(:,:,ndx), [28*28 length(ndx)]))';
ndxError = [];
classConf = zeros(10,10);
for i=1:Ntest
  y = mnist.test_labels(ndx(i));
  if doPlot
    figure(1);clf
    imagesc(mnist.test_images(:,:,ndx(i))); colormap(gray)
    title(sprintf('true class = %d', y))
  end
  for c=1:10
    loglik(i,c) = computeLoglik(Xtest(i,:), model(c));
  end
  %post(i,:) = exp(loglik(i,:)-logsumexp(loglik(i,:)));
  yhat= argmax(loglik(i,:))-1;
  if yhat ~= y
    nerr = nerr + 1;
    ndxError = [ndxError i];
  end
  classConf(y+1,yhat+1) = classConf(y+1,yhat+1)+1;
  if doPlot
    figure(2);clf
    bar(-loglik(i,:))
    set(gca,'xticklabel',0:9)
    ylabel('NLL')
    title(sprintf('test case %d, best guess is  %d', i, yhat))
    pause
  end
end
fprintf('Ntrain=%d, %d errors in %d, rate = %5.3f\n', ...
	Ntrain, nerr, Ntest, nerr/Ntest);
errorRate(trial) = nerr/Ntest;
classConf

for c=1:10
  fprintf('%3d & ', classConf(c,1:9));
  fprintf('%3d \\\\ \n', classConf(c,10));
  %fprintf('\n');
end

end

figure(3);clf;
plot(trainSize, errorRate, 'o-')
xlabel('training set size')
ylabel('error rate')
title('IID classifier on binarized MNIST digits')

keyboard

%%%%%%%%%%%%%

function ll = computeLoglik(vec, model)

% p(x) = prod_j theta(j)^I(xj=1) (1-theta(j))^I(xj=0)
% ll = sum_k I(xj=1) log theta(j) + I(xj=0) log (1-theta(j))
bitmask = vec > model.mu;
theta = model.pOn;
ll = sum(bitmask .* log(theta) + (1-bitmask) .* log(1-theta));

  
%%%%%%%%%%%%%

function [pOn, m] = trainModel(mnist, digit, Ntrain)

ndx = find(mnist.train_labels==digit);
ndx = ndx(1:Ntrain);
Xtrain = double(reshape(mnist.train_images(:,:,ndx), [28*28 length(ndx)]))';
% Xtrain is 1000 x 784

% binarize
m = mean(Xtrain(:));
Xtrain = double(Xtrain>m);

% Fit model
Non = sum( Xtrain==1, 1);
Noff = sum( Xtrain==0, 1);
a = 1; b = 1; % Laplace smoothing
pOn = (Non + a) ./ (Non + Noff + a + b); % posterior mean
