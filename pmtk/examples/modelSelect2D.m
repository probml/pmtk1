%% 2D Model Selection (Searching Over a Grid of Values)
% In this example, we carry on from where we left off in modelSelect1D and
% search over a 2d grid of values for a lambda-degree pair. Lambda is the L2
% regularizer and degree is the degree of the polynomial expansion of the data. 
% Most of what we said in the previous example applies here as well. The only
% difference is that our functions will be slightly more complicated. 
%% Model Space
% Again we make use of the the makeModelSpace static function to generate our 2d
% grid of points in the right format. Every combination of lambda and degree will
% be tried. 
load servo
verbose = true;
models = ModelSelection.makeModelSpace(logspace(-5,5,30) , 2:5 );
%% Test Function
% Our test function is slightly more complicated here. We see that it takes as
% parameters both lambda and degree. The built in scoring function will pass both
% of these values. We could also create this as a stand alone function or
% subfunction and create a handle to it, rather than create an anonymous function
% as we do here. If we were searching over an n-dimensional grid, using the
% default cv scoring function, predictFunction would take in d+3 parameters: 3 for
% Xtrain,ytrain,Xtest and one for each dimension. 
predictFunction = @(Xtrain,ytrain,Xtest,lambda,degree)...
  mode(predict(fit(LinregDist('transformer',...
  ChainTransformer({StandardizeTransformer(false),PolyBasisTransformer(degree)})),...
  'X',Xtrain,'y',ytrain,'prior','l2','lambda',lambda),Xtest));
%% 2D CV (MSE Loss)
% We run the model selection as before. Note that this time, it plots the error
% surface in 2D.
modelSelector = ModelSelection(          ...
    'predictFunction' , predictFunction       ,...
    'models'       , models             ,...
    'Xdata'        , Xtrain             ,...
    'verbose'      , verbose              ,...
    'Ydata'        , ytrain             );
bestLambda = modelSelector.bestModel{1};
bestSigma  = modelSelector.bestModel{2};
xlabel('lambda'); ylabel('degree');
title(sprintf('CV (MSE loss)\nchosen lambda: %f\nchosen degree: %f',bestLambda,bestSigma));
%%
% Assess performance
%%
T = ChainTransformer({StandardizeTransformer(false),PolyBasisTransformer(bestSigma)});
m = LinregDist('transformer',T);
m = fit(m,'X',Xtrain,'y',ytrain,'prior','l2','lambda',bestLambda);
CVmseError = mse(ytest,mode(predict(m,Xtest)))
%% 2D CV (NLL Loss)
% Our test function is almost the same here but we simply return the fitted
% object and not predictions. 
predictFunction = @(Xtrain,ytrain,Xtest,lambda,degree)...
  fit(LinregDist('transformer',...
  ChainTransformer({StandardizeTransformer(false),PolyBasisTransformer(degree)})),...
 'X',Xtrain,'y',ytrain,'prior','l2','lambda',lambda);
%% Loss Function
% Our loss function is the same as in the 1D CV NLL case. 
lossFunction = @(fittedObj,Xtest,ytest)-logprob(fittedObj,Xtest,ytest);
%%
modelSelector = ModelSelection(             ...
    'predictFunction' , predictFunction    ,...
    'lossFunction'    , lossFunction       ,...
    'models'          , models             ,...
    'Xdata'           , Xtrain             ,...
    'Ydata'           , ytrain             ,...
    'verbose'         , verbose              );
bestLambda = modelSelector.bestModel{1};
bestSigma  = modelSelector.bestModel{2};
xlabel('lambda'); ylabel('degree');
title(sprintf('CV (NLL loss)\nchosen lambda: %f\nchosen degree: %f',bestLambda,bestSigma));
%%
% Assess performance
%%
T = ChainTransformer({StandardizeTransformer(false),PolyBasisTransformer(bestSigma)});
m = LinregDist('transformer',T);
m = fit(m,'X',Xtrain,'y',ytrain,'prior','l2','lambda',bestLambda);
CVnllError = mse(ytest,mode(predict(m,Xtest)))
%% 2D BIC (Scoring Function)
% Much like the 1D case, we only need to create a custom scoring function.
% Notice that the candidate lambda value is available in model{1} and degree in
% model{2} since this is the order in which we created the model space. 
scoreFcn = @(obj,model)...
    -bicScore(fit(LinregDist('transformer',...
    ChainTransformer({StandardizeTransformer(false),PolyBasisTransformer(model{2})}))...
    ,'X',Xtrain,'y',ytrain,'prior','l2','lambda',model{1}),Xtrain,ytrain,model{1});
%%
modelSelector = ModelSelection(           ...
    'scoreFunction' , scoreFcn           ,...
    'models'        , models             ,...
    'Xdata'         , Xtrain             ,...
    'verbose'       , verbose              ,...
    'Ydata'         , ytrain             );
bestLambda = modelSelector.bestModel{1};
bestSigma  = modelSelector.bestModel{2};
xlabel('lambda'); ylabel('degree');
title(sprintf('BIC\nchosen lambda: %f\nchosen degree: %f',bestLambda,bestSigma));
%%
T = ChainTransformer({StandardizeTransformer(false),PolyBasisTransformer(bestSigma)});
m = LinregDist('transformer',T);
m = fit(m,'X',Xtrain,'y',ytrain,'prior','l2','lambda',bestLambda);
BicError = mse(ytest, mode(predict(m,Xtest)))
%% 2D AIC
% The steps to perform AIC are almost identical. 
scoreFcn = @(obj,model)...
    -aicScore(fit(LinregDist('transformer',...
    ChainTransformer({StandardizeTransformer(false),PolyBasisTransformer(model{2})}))...
    ,'X',Xtrain,'y',ytrain,'prior','l2','lambda',model{1}),Xtrain,ytrain,model{1});
%%
modelSelector = ModelSelection(           ...
    'scoreFunction' , scoreFcn           ,...
    'models'        , models             ,...
    'Xdata'         , Xtrain             ,...
    'verbose'       , verbose              ,...
    'Ydata'         , ytrain             );
bestLambda = modelSelector.bestModel{1};
bestSigma  = modelSelector.bestModel{2};
xlabel('lambda'); ylabel('degree');
title(sprintf('AIC\nchosen lambda: %f\nchosen degree: %f',bestLambda,bestSigma));
%%
T = ChainTransformer({StandardizeTransformer(false),PolyBasisTransformer(bestSigma)});
m = LinregDist('transformer',T);
m = fit(m,'X',Xtrain,'y',ytrain,'prior','l2','lambda',bestLambda);
AicError = mse(ytest,mode(predict(m,Xtest)))
%%