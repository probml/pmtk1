%% Logistic Regression: L1 "Support Vectors" 
% In this example, we fit a logistic regression model to synthetic data using an
% RBF expansion and an L1 regularizer. We visualize the "support vectors", i.e.
% the examples corresponding to non-zero weights. 
%
%% Load the Data
% Our synthetic data consists of 100 2D examples from two different classes, 1
% and 2. 
load synthetic2DdataSmall
%% Create the Data Transformer
% As in our previous example, we create transformer objects to preprocess our
% data and perform the RBF expansion. 
sigma2 = 1;          % kernel bandwidth
T = chainTransformer({standardizeTransformer(false)      ,...
                      kernelTransformer('rbf',sigma2)} );
%% Create the Model
% We create a new model by calling the constructor. 
model = logregDist('nclasses',2, 'transformer', T);
%% Fit the Model
% This time we use an L1 sparsity promoting prior. 
lambda = 0.001;                                              % L1 regularizer
model = fit(model,'prior','l1','lambda',lambda,'X',X,'y',Y);
%% Inspect the Parameters
% We now inspect the weight vector and find the training examples that
% correspond to the non-zero weights. We indicate these with circles on the
% figure.
supportVectors = X(model.w ~= 0,:);
%% Predict
% We will plot the decision boundary: the points that are equally likely to
% belong to each class. To do so, we must predict the class labels of a grid of
% points as in the previous example. 
[X1grid, X2grid] = meshgrid(0:0.01:1,0:0.01:1);
[nrows,ncols] = size(X1grid);
testData = [X1grid(:),X2grid(:)];
%%
% We extract the probabilities of each test point belonging to class 1
% and reshape the vector for plotting purposes. 
pred = predict(model,testData);   % pred is an object - a discrete distribution
pclass1 = pred.probs(:,1);                   
probGrid = reshape(pclass1,nrows,ncols);
%% Plot 
figure;
plot(X(Y==1,1),X(Y==1,2),'xr','MarkerSize',7,'LineWidth',1.5); hold on;
plot(X(Y==2,1),X(Y==2,2),'xb','MarkerSize',7,'LineWidth',1.5);
set(gca,'XTick',0:0.5:1,'YTick',0:0.5:1);
hold on;
plot(supportVectors(:,1),supportVectors(:,2),'ok','MarkerSize',10,'LineWidth',2)
contour(X1grid,X2grid,probGrid,'LineColor','k','LevelStep',0.5,'LineWidth',2);
title('L1 "Support Vectors"');
%%