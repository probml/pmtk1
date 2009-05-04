%% Classify the Mnist Digits Using KNN
% Takes about 3 minutes to run and gives an error rate of 2.31%
%% Load Data
%#broken - only predicting 4s and 5s
load mnistALL;
trainndx = 1:60000; testndx =  1:10000;
ntrain = length(trainndx);
ntest = length(testndx);
Xtrain = double(reshape(mnist.train_images(:,:,trainndx),28*28,ntrain)');
Xtest  = double(reshape(mnist.test_images(:,:,testndx),28*28,ntest)');
ytrain = (mnist.train_labels(trainndx));
ytest  = (mnist.test_labels(testndx));
clear mnist;
%% Create the Model
% Here we specify a class prior, use a local gaussian kernel and perform PCA
% dimensionality reduction by specifying a data transformer object. All of these
% are optional. 
classPrior = DirichletDist(0.05*normalize(1+histc(ytrain,unique(ytrain))));
model = KnnDist('K',3,'localKernel','gaussian','classPrior',classPrior,'beta',0.5,...
  'transformer',PcaTransformer('-k',60));
%% Fit and Predict
model = fit(model,Xtrain,ytrain);
clear Xtrain ytrain
pred = predict(model,Xtest);
err  = mean(ytest ~= mode(pred))