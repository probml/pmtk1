%% Classify Mnist Digits Using a Naive Bayes Classifier

binary = true;
Ntrain = 5000;
Ntest  = 1000;
[Xtrain,Xtest,ytrain,ytest] = setupMnist(binary, Ntrain, Ntest);
Xtrain = double(Xtrain); Xtest = double(Xtest);
classConditionals = copy(DiscreteDist('-support',[0,1]),1,10);
classPrior = DiscreteDist('-T',normalize(ones(10,1)),'-support',0:9);
classifier = GenerativeClassifierDist('classConditionals',classConditionals,'classPrior',classPrior);
classifier = fit(classifier,'X',Xtrain,'y',ytrain);
pred       = predict(classifier,Xtest);
yhat       = mode(pred);
err        = mean(yhat ~= ytest)

