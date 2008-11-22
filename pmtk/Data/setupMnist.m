function [Xtrain,Xtest,ytrain,ytest] = setupMnist(binary)

if(nargin == 0),binary = false;end


    
    
    
load mnistAll
Xtrain = (reshape(mnist.train_images,28*28,60000)');
Xtest = (reshape(mnist.test_images,28*28,10000)');
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