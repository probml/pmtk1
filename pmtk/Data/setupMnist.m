function [Xtrain,Xtest,ytrain,ytest] = setupMnist(binary)

if(nargin == 0),binary = false;end


    
    
    
load mnistAll
Xtrain = double(reshape(mnist.train_images,28*28,60000)');
Xtest = double(reshape(mnist.test_images,28*28,10000)');
ytrain = double(mnist.train_labels);
ytest = double(mnist.test_labels);
if(binary)
    mu = mean([Xtrain(:);Xtest(:)]);
    Xtrain = Xtrain >=mu;
    Xtest = Xtest >=mu;
end

clear mnist;

end