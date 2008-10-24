% Classify the MNIST digits using a one nearest neighbour classifier.
%
% Code by Matthew Dunham
%% Load Data
tic
load mnistAll;
if 0
  trainndx = 1:60000; testndx =  1:10000;
else
  trainndx = 1:10000; 
  testndx = 1:1000; 
end
ntrain = length(trainndx);
ntest = length(testndx);
Xtrain = double(reshape(mnist.train_images(:,:,trainndx),28*28,ntrain)');
Xtest  = double(reshape(mnist.test_images(:,:,testndx),28*28,ntest)');
% If you have the memory, removing 'sparse' will increase speed by about 25%
if 0
  Xtrain = sparse(Xtrain);
  Xtest = sparse(Xtest);
end

ytrain = (mnist.train_labels(trainndx));
ytest  = (mnist.test_labels(testndx));
clear mnist;
%% Precompute
XtrainSOS = sum(Xtrain.^2,2);
XtestSOS  = sum(Xtest.^2,2);
%% Setup
% fully vectorized solution takes too much memory so we will classify in batches
nbatches = 100;  % must be an even divisor of ntest, increase if you run out of memory
batches = mat2cell(1:ntest,1,(ntest/nbatches)*ones(1,nbatches));
ypred = zeros(ntest,1);
wbar = waitbar(0,sprintf('%d of %d classified',0,ntest));
%% Classify
for i=1:nbatches
    t = toc; waitbar(i/nbatches,wbar,sprintf('%d of %d Classified\nElapsed Time: %.2f seconds',(i-1)*(ntest/nbatches),ntest,t));
    dst = bsxfun(@plus,XtestSOS(batches{i},:)',XtrainSOS) - 2*Xtrain*Xtest(batches{i},:)'; %compute Euclidean distance
    [junk,closest] = min(dst,[],1);
    ypred(batches{i}) = ytrain(closest);
end
%% Report
close(wbar);
errorRate = mean(ypred ~= ytest);
fprintf('Error Rate: %.2f%%\n',100*errorRate);
t = toc; fprintf('Total Time: %.2f seconds\n',t);
%%
