%% KNN Mnist Error Rate W.R.T. N
% 
% In this example, we plot the classification error rate on the mnist data set
% as we vary the size of the training set. 
%% Load Data
% We will test on the same 1000 randomly selected examples and use the remaining
% 69000 examples as our training base.
%%
run = false;  % set to true to regenerate the results
if(run)
load mnistALL;
setSeed(0);
ntest  = 1000;
Xtrain = sparse(double(reshape(mnist.train_images,28*28,[])'));
Xtest  = sparse(double(reshape(mnist.test_images ,28*28,[])'));
ytrain = mnist.train_labels;
ytest  = mnist.test_labels;
clear mnist;
perm   = randperm(size(Xtest,1));
Xtest  = Xtest(perm,:);
ytest  = ytest(perm,:);
Xtrain = [Xtrain;Xtest(ntest+1:end,:)];
ytrain = [ytrain;ytest(ntest+1:end,:)];
Xtest  = Xtest(1:ntest,:);
ytest  = ytest(1:ntest,:);
perm   = randperm(69000);
Xtrain = Xtrain(perm,:);
ytrain = ytrain(perm,:);
ntrain = [100,1000,2000,5000,10000:10000:60000,69000];
%% Run KNN
errors = zeros(size(ntrain));
for i=1:numel(ntrain)
    model     = fit(KnnDist('K',1),Xtrain(1:ntrain(i),:),ytrain(1:ntrain(i)));
    pred      = predict(model,Xtest);
    errors(i) = mean(ytest ~= mode(pred));
end
clearvars -except ntrain errors ntest
save mnist1NNresults
else
   load mnist1NNresults; 
end
%% Plot Results
figure; hold on;
plot(ntrain,errors,'-b','LineWidth',5);
plot(ntrain,errors,'.r','MarkerSize',30);
box on; grid on;
xlabel('Training Set Size');
ylabel('Error Rate');
title(sprintf('MNIST 1NN Error Rate vs Training Size\n Fewest Errors = %d / %d',ntest*min(errors),ntest));
%%

