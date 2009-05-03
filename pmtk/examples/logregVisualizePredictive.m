%% Logistic Regression: Visualizing the Predictive Distribution
% Here we fit a binary logistic regression model to synthetic data and visualize the
% predictive distribution. We compare the MLE to L1 and L2 regularized
% models.
%% Load and Plot the Data
% Load synthetic data generated from a mixture of Gaussians. Source:
% <<http://research.microsoft.com/~cmbishop/PRML/webdatasets/datasets.htm>>
%
load bishop2class
figure;
plot(X(Y==1,1),X(Y==1,2),'xr','LineWidth',2,'MarkerSize',7); hold on;
plot(X(Y==2,1),X(Y==2,2),'xb','LineWidth',2,'MarkerSize',7);
D = DataTable(X,Y);
title('Training Data');
legend({'Class1','Class2'},'Location','BestOutside');

%% Cross Validation
% First we find the optimal hyper-parameters using CV
if 0
  [lambdaL2, sigmaL2, lambdaL1, sigmaL1] = logregModelSel2d();
else
  % For speed, we just hard-code the results
  lambdaL2 = 10;
  sigmaL2 = 0.3;
  lambdaL1 = 2;
  sigmaL1 =  0.4;
  
  lambdaL1 = 2.1544;
  sigmaL1 = 0.2;
end

%{
%% Create the model
% We will make use of PMTK's transformer objects to easily preprocess the data
% and perform the basis expansion. We chain 2 transformers together, which
% will be applied to the data in sequence. When we pass our ChainTransformer to
% our model, (which we will create shortly), all of the details of the
% transformation are retained, and where appropriate, applied to future test data.
%
T = ChainTransformer({StandardizeTransformer(false)      ,...
    KernelTransformer('rbf',sigmaL2)} );
%
% We now create a new logistic regression model and pass it the transformer object
% we just created.
% We specify that the y labels are from {1,2} (as opposed to, say, {0,1})
model = LogregBinaryL2('-lambda', lambdaL2, '-labelSpace', [1,2], '-transformer', T);

%% Fit the Model
% We now find a MAP estimate of the parameters
% We can choose the optimizer by changing model.optMethod
model = fit(model,D);

%% Predict using plugin
% To visualize the predictive distribution we will first create grid of points
% in our original 2D feature space and evaluate the posterior probability that
% each point belongs to class 1.
%
[X1grid, X2grid] = meshgrid(-3:0.02:3,-3:0.02:3);
[nrows,ncols] = size(X1grid);
testData = [X1grid(:),X2grid(:)];
[yhat, pred] = predict(model,testData); % pred is a Bernoulli
pclass1 = pmf(pred); % table of probabilities
probGrid = reshape(pclass1,nrows,ncols);
%% Plot the Predictive Distribution
% We can now make use of Matlab's excellent plotting capabilities and plot the
% surface of the distribution. Notice the relatively smooth decision
% boundary. This is due in large part to the value of Sigma. We will see
% shortly what happens when we use a relatively small value.
figure; hold on;
surf(X1grid,X2grid,probGrid);
shading interp; view([0 90]); colorbar;
alpha 0.8;
box on;
contour(X1grid,X2grid,probGrid,'LineColor','k','LevelStep',0.5,'LineWidth',2.5);
title('Predictive Distribution (L2 Logistic Regression)');
%% Plot Decision Boundary
% We can plot the decision boundary along with the data
figure; hold on;
plot(X(Y==1,1),X(Y==1,2),'xr','LineWidth',2,'MarkerSize',7); hold on;
plot(X(Y==2,1),X(Y==2,2),'xb','LineWidth',2,'MarkerSize',7);
title('Decision Boundary (L2 Logistic Regression)');
box on;
contour(X1grid,X2grid,probGrid,'LineColor','k','LevelStep',0.5,'LineWidth',2.5);
%}

%% L1 Prior
% Now lets use an L1 prior and refit the model
T = ChainTransformer({StandardizeTransformer(false)      ,...
    KernelTransformer('rbf',sigmaL1)} );
model = LogregL1('-lambda', lambdaL1, '-labelSpace',[1,2], ...
  '-transformer', T, '-verbose', false, '-addOnes', false); %'-optMethod', 'iteratedRidge');
model = fit(model,D);

[X1grid, X2grid] = meshgrid(-3:0.02:3,-3:0.02:3);
[nrows,ncols] = size(X1grid);
testData = [X1grid(:),X2grid(:)];
[yhat, pred] = predict(model,testData);              
pclass01 = pmf(pred);
pclass1 = pclass01(1,:);
probGrid = reshape(pclass1,nrows,ncols);
%% Plot the Predictive Distribution L1
figure; hold on;
surf(X1grid,X2grid,probGrid);
shading interp; view([0 90]); colorbar;
alpha 0.8;
box on;
contour(X1grid,X2grid,probGrid,'LineColor','k','LevelStep',0.5,'LineWidth',2.5);
title('Predictive Distribution (L1 Logistic Regression)');
%% Plot Decision Boundary L1
% We can plot the decision boundary along with the data
figure; hold on;
plot(X(Y==1,1),X(Y==1,2),'xr','LineWidth',2,'MarkerSize',7); hold on;
plot(X(Y==2,1),X(Y==2,2),'xb','LineWidth',2,'MarkerSize',7);
title('Decision Boundary (L1 Logistic Regression)');
box on;
contour(X1grid,X2grid,probGrid,'LineColor','k','LevelStep',0.5,'LineWidth',2.5);
%% Identify "support vectors"
% We now visualize the "support vectors", i.e.
% the examples corresponding to non-zero weights.
% Since we are using an RBF kernel, feature j corresponds to example j.
supportVectors = X(abs(model.w) > 1e-5,:);
plot(supportVectors(:,1),supportVectors(:,2),'ok','MarkerSize',10,'LineWidth',2)

