function [Xtrain,Xtest,ytrain,ytest] = setupMnist(binary, Ntrain, Ntest)

if nargin < 1, binary = false; end
if nargin < 2, Ntrain = 60000; end
if nargin < 3, Ntest = 10000; end
  
load mnistAll
Xtrain = reshape(mnist.train_images(:,:,1:Ntrain),28*28,Ntrain)';
Xtest = reshape(mnist.test_images(:,:,1:Ntest),28*28,Ntest)';
ytrain = (mnist.train_labels);
ytest = (mnist.test_labels);
clear mnist;
if(binary)
    mu = mean([Xtrain(:);Xtest(:)]);
    Xtrain = Xtrain >=mu;
    Xtest = Xtest >=mu;
end
ytrain = double(ytrain);
ytest  = double(ytest);

end