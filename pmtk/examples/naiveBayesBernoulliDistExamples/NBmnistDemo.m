%% Naive Bayes (Bernoulli) on the MNIST Data Set
% In this example, we apply Naive Bayes to the MNIST data set. We assess
% generalization performance, plot the class confusion matrix and display
% samples from the joint distribution. 
%% Load Data
% We convert the images to binary.
[Xtrain,Xtest,ytrain,ytest] = setupMnist(true);
%% Create the Model
nb = NaiveBayesBernoulliDist('nclasses',10);
%% Priors
% We specify a prior over class labels, a Dirichlet distribution as well as a
% feature prior. If these are not specified, uninformative priors are used
% instead. 
classPrior   = DirichletDist(2*ones(1,10));
featurePrior = BetaDist(2,2);
%% Fit
nb = fit(nb,'X',Xtrain,'y',ytrain,'featurePrior',featurePrior,'classPrior',classPrior);
%% Predict
%  Here pred is a DiscreteDist object. 
pred = predict(nb,Xtest);
%% Assess Performance
yhat = mode(pred);
err = mean(yhat~=ytest)
%% Plot Samples We sample
% We sample from the posterior, once for each value of y, 0:9, and visualize
% these as images. Note the color information is not present in the samples. 
for i=0:9
    figure
    imagesc(reshape(sample(nb,i),28,28));
    set(gca,'XTick',[],'YTick',[]);
end
%% Plot the Class Confusion Matrix
% We plot the class confusion matrix as a hinton diagram. 
ccm = zeros(10,10);
for i=1:10
    for j=1:10
        ccm(i,j) = sum(yhat == i-1 & ytest == j-1);
    end
end
hintonDiagram(ccm);
title('Class Confusion Matrix');
xlabel('actual','FontSize',12);
ylabel('predicted','FontSize',12);
labels = {'0','1','2','3','4','5','6','7','8','9'};
set(gca,'XTickLabel',labels,'YTicklabel',labels,'box','on','FontSize',12);