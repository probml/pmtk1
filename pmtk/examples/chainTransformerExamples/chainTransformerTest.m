%% Test the Chain Transformer
% Note, it is not usually necessary to instantiate transformers directly, as in
% this example. If you specify a transformer to a model, this will be taken care
% of automatically. 
%#testPMTK
T = ChainTransformer({StandardizeTransformer, AddOnesTransformer});
setSeed(0);
Xtrain = rand(5,3);
Xtest = rand(5,3);
[Xtrain1, T]= train(T, Xtrain);
[Xtest1] = test(T, Xtest);

ntrain = size(Xtrain, 1); ntest = size(Xtest, 1);
mu = mean(Xtrain); s = std(Xtrain);
Xtrain2 = Xtrain - repmat(mu, ntrain, 1);
Xtrain2 = Xtrain2 ./ repmat(s, ntrain, 1);
Xtrain2 = [ones(ntrain,1) Xtrain2];
assert(approxeq(Xtrain1, Xtrain2))

Xtest2 = Xtest - repmat(mu, ntest, 1);
Xtest2 = Xtest2 ./ repmat(s, ntest, 1);
Xtest2 = [ones(ntest,1) Xtest2];
assert(approxeq(Xtest1, Xtest2))