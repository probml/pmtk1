%% Logistic Regression: Visualizing the Predictive Distribution
% Here we fit a logistic regression model to synthetic data and visualize the
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
title('Training Data');
legend({'Class1','Class2'},'Location','BestOutside');
%% Cross Validate L2
% Here we cross validate sigma and lambda simultaneously for L2 LR. This
% takes about 3 minutes to run.
if(0)
    %%
    % We create our test function. See the ModelSelection class for more
    % details.
    testFunction = @(Xtrain,ytrain,Xtest,lambda,sigma)...
        mode(predict(fit(LogregDist('nclasses',2,'transformer',...
        ChainTransformer({StandardizeTransformer(false),...
        KernelTransformer('rbf', sigma)})),...
        'X',Xtrain,'y',ytrain,'priorStrength',lambda,'prior','l2'),Xtest));
    %%
        % This is the range we will search over; every combination will be
        % tested.
        lambdaRange = logspace(-2,1.5,30);
        sigmaRange = 0.1:0.1:2;
        modelSpace = ModelSelection.makeModelSpace(lambdaRange,sigmaRange);
        %%
        % Finally we perform the model selection.
        mselect = ModelSelection(                ...
            'testFunction' , testFunction        ,...
            'models'       , modelSpace          ,...
            'Xdata'        , X                   ,...
            'Ydata'        , Y                   );
        lambdaL2 = mselect.bestModel{1};
        sigmaL2  = mselect.bestModel{2};
else
    %%
    % To save time, here are the results of the cross validation.
    %sigma = 2;
    %lambda = 0.0078476;
    lambdaL2 = 7.8805;
    sigmaL2 = 0.2;
end
% We are now ready to fit.
%% Create the Data Transformer
% We will make use of PMTK's transformer objects to easily preprocess the data
% and perform the basis expansion. We chain three transformers together, which
% will be applied to the data in sequence. When we pass our ChainTransformer to
% our model, (which we will create shortly), all of the details of the
% transformation are retained, and where appropriate, applied to future test data.
%
T = ChainTransformer({StandardizeTransformer(false)      ,...
    KernelTransformer('rbf',sigmaL2)} );
%% Create the Model
% We now create a new logistic regression model and pass it the transformer object
% we just created.
model = LogregDist('nclasses',2, 'transformer', T);
%% Fit the Model
% To fit the model, we simply call the model's fit method and pass in the data.
% Here we use an L2 regularizer, however, an L1 sparsity promoting regularizer
% could have been used just as easily by replacing the string 'l2' with
% 'l1' as we will see later. We are performing map estimation here,
% however in the L2 case we can perform full Bayesian estimation by
% appending 'method','bayesian' to the call to fit().
model = fit(model,'prior','l2','priorStrength',lambdaL2,'X',X,'y',Y);
%%
% We can specify which optimization method we would like to use by passing in
% its name to the fit method as in the following. There are number of options
% but reasonable defaults exist.
%%
%  model = fit(model,'prior','l2','lambda',lambda,'X',X,'y',Y,'optMethod','lbfgs');
%% Predict
% To visualize the predictive distribution we will first create grid of points
% in our original 2D feature space and evaluate the posterior probability that
% each point belongs to class 1. We are using the map as a plugin
% approximation. If we had fit with 'method','bayesian', we could also
% have predicted by drawing Monte Carlo samples and averaging using
% 'method','mc'.
%
[X1grid, X2grid] = meshgrid(-3:0.02:3,-3:0.02:3);
[nrows,ncols] = size(X1grid);
testData = [X1grid(:),X2grid(:)];
%%
% The output of the predict method is a discrete distribution over the class
% labels. We extract the probabilities of each test point belonging to class 1
% and reshape the vector for plotting purposes.
pred = predict(model,testData);              % pred is an object - a discrete distribution
pclass1 = pred.mu(1,:)';
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
%% L1 Prior
% Now lets use an L1 prior and repeat our steps. 
if(0) % Takes about 3 minutes

     testFunction = @(Xtrain,ytrain,Xtest,lambda,sigma)...
        mode(predict(fit(LogregDist('nclasses',2,'transformer',...
        ChainTransformer({StandardizeTransformer(false),...
        KernelTransformer('rbf', sigma)})),...
        'X',Xtrain,'y',ytrain,'priorStrength',lambda,'prior','l1'),Xtest));

    lambdaRange = logspace(-1,1,10);
    sigmaRange = [0.2,0.5:0.5:4];
    modelSpaceL1 = ModelSelection.makeModelSpace(lambdaRange,sigmaRange);
    mSelectL1 = ModelSelection(                 ...
        'testFunction' , testFunction            ,...
        'models'     , modelSpaceL1              ,...
        'Xdata'        , X                       ,...
        'Ydata'        , Y                       );
    lambdaL1 = mSelectL1.bestModel{1};
    sigmaL1 = mSelectL1.bestModel{2};
else
    %%
    % To save time, here are the results of the cross validation.
    %lambdaL1 = 1.2915;
    lambdaL1 = 2.1544;
    sigmaL1 = 0.2;
end
T = ChainTransformer({StandardizeTransformer(false)      ,...
    KernelTransformer('rbf',sigmaL1)} );
model = LogregDist('nclasses',2, 'transformer', T);
model = fit(model,'prior','l1','priorStrength',lambdaL1,'X',X,'y',Y);
[X1grid, X2grid] = meshgrid(-3:0.02:3,-3:0.02:3);
[nrows,ncols] = size(X1grid);
testData = [X1grid(:),X2grid(:)];
pred = predict(model,testData);              % pred is an object - a discrete distribution
pclass1 = pred.mu(1,:)';
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
supportVectors = X(model.w ~= 0,:);
plot(supportVectors(:,1),supportVectors(:,2),'ok','MarkerSize',10,'LineWidth',2)
%% MLE with Small Sigma
% Here we investigate what happens when we use the MLE and a small value
% for sigma. The decision boundary becomes much more complex but of
% course we are overfitting.
lambda = 0;                 % lambda = 0 corresponds to MLE
sigma = 0.5;                % arbitrarily chosen small value for sigma
T = ChainTransformer({StandardizeTransformer(false),KernelTransformer('rbf',sigma)} );
model = LogregDist('nclasses',2, 'transformer', T);
model = fit(model,'X',X,'y',Y);
[X1grid, X2grid] = meshgrid(-3:0.02:3,-3:0.02:3);
[nrows,ncols] = size(X1grid);
testData = [X1grid(:),X2grid(:)];
pred = predict(model,testData);
pclass1 = pred.mu(1,:)';
probGrid = reshape(pclass1,nrows,ncols);
figure; hold on;
surf(X1grid,X2grid,probGrid);
shading interp; view([0 90]); colorbar;
alpha 0.8;
box on;
title('Predictive Distribution (MLE & Small Sigma)');
figure; hold on;
plot(X(Y==1,1),X(Y==1,2),'xr','LineWidth',2,'MarkerSize',7); hold on;
plot(X(Y==2,1),X(Y==2,2),'xb','LineWidth',2,'MarkerSize',7);
title('Decision Boundary (MLE & Small Sigma)');
box on;
contour(X1grid,X2grid,probGrid,'LineColor','k','LevelStep',0.5,'LineWidth',2.5);