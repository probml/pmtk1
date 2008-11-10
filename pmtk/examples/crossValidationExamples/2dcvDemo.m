%% Cross Validation Over a 2D Grid of Values
% Here we demonstrate how to cross validate two values, lambda and sigma
% simultaneously using the CrossValidation class. We use the crabs data set and
% perform l2 logistic regression with an RBF expansion.
%%
% This model selection class requires that two key functions be specified: a loss
% function, such as zero-one loss, mean squared error, nll, etc, and test function
% of the following form:
%%
%  @(Xtrain,ytrain,Xtest,lambda,sigma)testFunction(Xtrain,ytrain,Xtest,lambda,sigma)
%%
% If we were cross validating over a single value, we would omit the "sigma". We
% can cross validate over an n-dimensional grid just as easily, e.g.
%%
%  @(Xtrain,ytrain,Xtest,lambda,sigma)testFunction(Xtrain,ytrain,Xtest,lambda,sigma,eta,gamma)
%%
% However, the problem soon becomes computationally intractable.
%
% The test function will be evaluated at each fold for each combination of
% values and its output will be passed directly to the loss function. In our
% example, we will use zero-one loss and thus our test function must the return
% predicted labels.
%% Create the Test Function
% Creating the test function will usually amount to composing fit and predict
% functions together. This is what we do here, however because of the number of
% options available and the RBF preprocessing, this will be a more advanced
% example.
%
%%
% One approach is to write a stand alone function with the right behavior. Here
% is an example.
%%
%  function yhat = testFunction(Xtrain,ytrain,Xtest,lambda,sigma)
%      T = ChainTransformer({StandardizeTransformer(false),KernelTransformer(sigma)});
%      m = LogregDist('nclasses',2,'transformer',T);
%      m = fit(m,'X',Xtrain,'y',ytrain,'lambda',lambda,'prior','l2');
%      pred = predict(m,Xtest);
%      yhat = mode(pred);
%  end
%
%%
% We would then create the function handle as follows:
% tfunc = @testFunction;
%
%% Use Anonymous Functions
% However, it is often convenient to create the test function on the fly using
% anonymous functions.
%
% The test function is the composition m( p( f( c( t( s , k ) ) ) ) where
%%
% * m is mode()
% * p is fit()
% * c is the model constructor
% * t is the chain transformer constructor
% * s is the StandardizeTransformer constructor
% * k is the kernalTransformer constructor
%%
% Our five input variables are defined in these functions as follows:
%%
% * f(Xtrain,ytrain,lambda)
% * p(Xtest)
% * k(sigma)
%%
% To make the composition clearer we will curry our functions, however,
% this is not strictly necessary.
%
%%
m = @mode;
p = @(model,Xtest)predict(model,Xtest);
f = @(model,Xtrain,ytrain,lambda)fit(model,'X',Xtrain,'y',ytrain,'lambda',lambda,'prior','l2');
c = @(trans)LogregDist('nclasses',2,'transformer',trans);
t = @(a,b)ChainTransformer({a(),b()});  % Use () to force evaluation before passing on
s = @(x)StandardizeTransformer(false);
k = @(sigma)KernelTransformer('rbf',sigma);
%%
% Now let us compose c,t,s,k
c = @(sigma)c(t(s(),k(sigma)));
%%
% Finally we compose the remaining functions.
testFunction = @(Xtrain,ytrain,Xtest,lambda,sigma)m(p(f(c(sigma),Xtrain,ytrain,lambda),Xtest));
%%
% Of course we could have done this all in one step.
%%
%  testFunction = @(Xtrain,ytrain,Xtest,lambda,sigma)...
%  mode(predict(fit(LogregDist(...
%  'nclasses',2,'transformer',...
%  ChainTransformer(...
%  {StandardizeTransformer(false),KernelTransformer('rbf', sigma)})),...
%  'X',Xtrain,'y',ytrain,'lambda',lambda,'prior','l2'),Xtest));
%
%% Create the Loss Function
% We are now ready to perform the cross validation. We pass in a cell
% array with test values for lambda and sigma and specify a loss function.
% MSE and Zero-One are built in and can be used by specifying a string
% instead of a function handle, such as 'MSE', or 'ZeroOne'. Here we use
% zero one loss but use a function handle for demonstration purposes.
%
%%
lossFunction = @(yhat,ytest)sum(reshape(yhat,size(ytest))~=ytest);
%% Perform the Cross Validation
load crabs;
%%
% Performing the actual cross validation simply amounts to instantiating the
% class with the right inputs.
modelSelection = CrossValidation(                     ...
    'testFunction' , testFunction                    ,...
    'CVvalues'     , { logspace(-5,0,20) , 1:0.5:15 },... % every combination will be tested
    'lossFunction' , lossFunction                    ,...
    'verbose'      , true                           ,... % true by default - shows progress
    'Xdata'        , Xtrain                          ,...
    'Ydata'        , ytrain                          );

bestVals = modelSelection.bestValue;
bestLambda = bestVals(1)                                  %#ok
bestSigma  = bestVals(2)                                  %#ok
set(gca,'XScale','log');
%% Plot Results
% By default, a figure of the cross validation curve is plotted, (at least in 2D
% and 3D). Here we transform the x-axis since our lambda values were log-spaced.
%%
%  set(gca,'XScale','log');
%% Refit
% Now lets retrain the model using the best lambda and sigma values
T = ChainTransformer({StandardizeTransformer(false),KernelTransformer('rbf',bestSigma)});
m = LogregDist('nclasses',2,'transformer',T);
m = fit(m,'X',Xtrain,'y',ytrain,'lambda',bestLambda,'prior','l2');
pred = predict(m,Xtest);
yhat = mode(pred);
errorRate = mean(yhat'~=ytest)
%%
% Notice that we correctly classify all of the examples.

