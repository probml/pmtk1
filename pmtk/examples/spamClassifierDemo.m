% Perform classification on spam demo, using dataset from Hastie, Tribshirani,and Friedman
%#author Cody Severinski

function [model,baseModel] = spamDemo()

tic;
% For reproducibility
setSeed(0);

% Read the data and permute
origData = load('spam.data');
[n] = size(origData,1);
origData = origData(randsample(n,n),:);
% spam = 1, email = 0
classLabel = origData(:,end);
% features
spam = origData(:,1:(end-1));
% Different versions of the data - log transformed, binary
logspam = log( spam + 0.1);
binspam = spam > 0;
binspam = double(binspam);

d = size(spam,2);

% K is the number of times we partition the training data
K = 5;
nModels = 6;

% lambda for l2-regularization
nLambda = 10;
lambda = logspace(-10,-1,nLambda);

% Get indices for each of the folds
[trainfolds,testfolds] = Kfold(n,K);

% A function handle to our custom score function.  This is needed to ensure that we use the same folds regardless of whether we use the ModelSelection class or not
cvCustom = @(obj,model)scoreFunction(obj,model);

modelName = {'Naive Bayes', 'Logistic Regression', 'Diagonal Gaussian', 'Logistic Regression', 'Laplace Approximation', 'Laplace Approximation'};
dataName = {'Binary', 'Binary', 'log-transformed', 'log-transformed', 'continuous', 'log-transformed'};

fprintf('Constructing Base models: %s, %s \n', modelName{1}, dataName{1});
% Naive Bayes
% Define discrete class conditionals, with support on [0,1], taking on one of two classes; Define equal probability priors
baseModel{1}.classConditionals = copy(DiscreteDist('support',[0,1]),1,2);
baseModel{1}.classPrior = DiscreteDist('mu',normalize(ones(2,1)),'support',0:1);
baseModel{1}.model = GenerativeClassifierDist('classConditionals',baseModel{1}.classConditionals,'classPrior',baseModel{1}.classPrior);
baseModel{1}.X = binspam;

fprintf('Constructing Base models: %s, %s \n', modelName{2}, dataName{2});
% Logistic regression on binary data
baseModel{2}.model = @(lambda)LogregDist('prior','l2','priorStrength',lambda);
baseModel{2}.modelspace = ModelSelection.makeModelSpace(lambda);
baseModel{2}.predictFunction = @(Xtrain,ytrain,Xtest,lambda)...
	mode(predict(fit(baseModel{2}.model(lambda),'X',Xtrain,'y',ytrain),Xtest));
baseModel{2}.X = binspam;
baseModel{2}.modelselection = ModelSelection(              ...
		  'predictFunction',baseModel{2}.predictFunction ,...      % the test function we just created
			'scoreFunction',cvCustom ,...
		  'Xdata'       ,baseModel{2}.X         ,...      % all of the X data we have available
		  'Ydata'       ,classLabel             ,...      % all of the y data we have available
		  'verbose'     ,true            ,...      % turn off progress report
		  'models'      ,baseModel{2}.modelspace					,...% the model space created above
			'CVnfolds'		, 5 );
baseModel{2}.lambda = baseModel{2}.modelselection.bestModel{1};
baseModel{2}.model = baseModel{2}.model(baseModel{2}.lambda);

fprintf('Constructing Base models: %s, %s \n', modelName{3}, dataName{3});
% Diagonal Gaussian on log transformed
% Define discrete class conditionals, with support on [0,1], taking on one of two classes
baseModel{3}.classConditionals = copy(MvnDist(1/2*ones(1,d),diag(0.3*ones(1,d)),'prior','nig','covtype','diagonal'),1,2);
% Define equal probability priors
baseModel{3}.classPrior = DiscreteDist('mu',normalize(ones(2,1)),'support',0:1);
baseModel{3}.model = GenerativeClassifierDist('classConditionals',baseModel{3}.classConditionals,'classPrior',baseModel{3}.classPrior);
baseModel{3}.X = logspam;

fprintf('Constructing Base models: %s, %s \n', modelName{4}, dataName{4});
% logistic regression using log transformed
baseModel{4}.model = @(lambda)LogregDist('prior','l2','priorStrength',lambda);
baseModel{4}.modelspace = ModelSelection.makeModelSpace(lambda);
baseModel{4}.predictFunction = @(Xtrain,ytrain,Xtest,lambda)...
   mode(predict(fit(baseModel{4}.model(lambda),'X',Xtrain,'y',ytrain),Xtest));
baseModel{4}.X = logspam;
baseModel{4}.modelselection = ModelSelection(              ...
		  'predictFunction',baseModel{4}.predictFunction ,...      % the test function we just created
			'scoreFunction',cvCustom ,...
		  'Xdata'       ,baseModel{4}.X          ,...      % all of the X data we have available
		  'Ydata'       ,classLabel             ,...      % all of the y data we have available
		  'verbose'     ,true            ,...      % turn off progress report
		  'models'      ,baseModel{4}.modelspace ,...
			'CVnfolds'		, 5					);        % the model space created above
baseModel{4}.lambda = baseModel{4}.modelselection.bestModel{1};
baseModel{4}.model = baseModel{4}.model(baseModel{4}.lambda);

fprintf('Constructing Base models: %s, %s \n', modelName{5}, dataName{5});
% now the laplace approximation to the continuous data
baseModel{5}.model = @(lambda)Logreg_MvnDist('infMethod','laplace','priorStrength',lambda);
baseModel{5}.modelspace = ModelSelection.makeModelSpace(lambda);
baseModel{5}.X = spam;
baseModel{5}.predictFunction = @(Xtrain,ytrain,Xtest,lambda)...
   mode(predict(fit(baseModel{5}.model(lambda),'X',Xtrain,'y',ytrain),Xtest));
baseModel{5}.modelselection = ModelSelection(              ...
		  'predictFunction',baseModel{5}.predictFunction ,...      % the test function we just created
			'scoreFunction',cvCustom ,...
		  'Xdata'       ,baseModel{5}.X          ,...      % all of the X data we have available
		  'Ydata'       ,classLabel             ,...      % all of the y data we have available
		  'verbose'     ,true            ,...      % turn off progress report
		  'models'      ,baseModel{5}.modelspace     ,...% the model space created above
			'CVnfolds'		, 5);        
baseModel{5}.lambda = baseModel{5}.modelselection.bestModel{1};
baseModel{5}.model = baseModel{5}.model(baseModel{5}.lambda);

fprintf('Constructing Base models: %s, %s \n', modelName{6}, dataName{6});
% now the laplace approximation to the continuous data
baseModel{6}.model = @(lambda)Logreg_MvnDist('infMethod','laplace','priorStrength',lambda);
baseModel{6}.modelspace = ModelSelection.makeModelSpace(lambda);
baseModel{6}.X = logspam;
baseModel{6}.predictFunction = @(Xtrain,ytrain,Xtest,lambda)...
   mode(predict(fit(baseModel{6}.model(lambda),'X',Xtrain,'y',ytrain),Xtest));
baseModel{6}.modelselection = ModelSelection(              ...
		  'predictFunction',baseModel{6}.predictFunction ,...      % the test function we just created
			'scoreFunction',cvCustom ,...
		  'Xdata'       ,baseModel{6}.X          ,...      % all of the X data we have available
		  'Ydata'       ,classLabel             ,...      % all of the y data we have available
		  'verbose'     ,true            ,...      % turn off progress report
		  'models'      ,baseModel{6}.modelspace     ,...% the model space created above
			'CVnfolds'		, 5);        
baseModel{6}.lambda = baseModel{6}.modelselection.bestModel{1};
baseModel{6}.model = baseModel{6}.model(baseModel{6}.lambda);

for mod=1:nModels
	fprintf('Fitting: %s, %s \n', modelName{mod}, dataName{mod});
	% Fit the naive bayes classifier
	for fold=1:K
		model{mod,fold}.classifier	= fit( baseModel{mod}.model,'X',baseModel{mod}.X(trainfolds{fold},:),'y',classLabel(trainfolds{fold}) );
		model{mod,fold}.predict			= predict( model{mod,fold}.classifier, baseModel{mod}.X(testfolds{fold},:) );
		model{mod,fold}.yhat				= mode( model{mod,fold}.predict );
		model{mod,fold}.err					= mean( model{mod,fold}.yhat ~= classLabel(testfolds{fold}) );
		[model{mod,fold}.FPrate,model{mod,fold}.TPrate,model{mod,fold}.auc, model{mod,fold}.threshold] = computeROC( model{mod,fold}.predict.mu(2,:),classLabel(testfolds{fold}) );
	end
end
% Get the errors

err = zeros(nModels,K);
auc = zeros(nModels,K);
for mod=1:nModels
	for fold=1:K
		err(mod,fold) = model{mod,fold}.err;
		auc(mod,fold) = model{mod,fold}.auc;
	end
end
errMean = mean(err,2);
errStd = std(err,1,2);
aucMean = mean(auc,2);
aucStd = std(auc,1,2);

fprintf('Model \t\t\t Data \t\t\t Mean Error \t Std Error \t Mean AUC \t Std AUC\n');
for mod=1:nModels
	fprintf('%s \t %s \t %g \t %g \t %g \t %g \n', modelName{mod}, dataName{mod}, errMean(mod), errStd(mod), aucMean(mod), aucStd(mod));
end

% Create plots of ROC Curves
for mod=1:nModels
foldidx = argmax(auc(mod,:));
figure(); hold on;
plot(model{mod,foldidx}.FPrate, model{mod,foldidx}.TPrate,'linewidth',3);
line([0,1],[1,0],'color','k','linewidth',3);
xlabel('false positive rate'); ylabel('true positive rate'); title(sprintf('ROC Curve: %s (%s Data)',modelName{mod},dataName{mod}));
pdfcrop; print_pdf(sprintf('roc-%s-%s',modelName{mod},dataName{mod}));
end

for mod=5:6
foldidx = argmax(auc(mod,:));
% Marginal posterior credible intervals
vn = length(trainfolds{foldidx});
mu = model{mod,foldidx}.classifier.wDist.mu;
Sigma = model{mod,foldidx}.classifier.wDist.Sigma;
credibles = [mu + tinv(0.025, vn)*sqrt(diag(Sigma)), mu + tinv(1-0.025, vn)*sqrt(diag(Sigma))];
figure(); hold on;
line([(1:d);(1:d)],[credibles(:,1)';credibles(:,2)'], 'color','k','linewidth',3);
line([0,d],[0,0],'color','r','linewidth',3);
title(sprintf('Marginal posterior credible intervals.  %s (%s Data)',modelName{mod},dataName{mod}));
xlabel('Magnitude'); ylabel('Feature index');
pdfcrop; print_pdf(sprintf('laplaceapprox-marginal-%s',num2str(mod)));
end







% Now compute mutual information
theta = [mean(binspam(classLabel == 0,:))', mean(binspam(classLabel == 1,:))'];
[mi,pw] = NBmi(theta);
toc;

% If the book will not be printed in color, then this is better 
%figure(); hold on;
%subplot(2,1,1);
%line([1:d;1:d],[zeros(size(1:d));theta(:,1)'],'color','k','linewidth',3)
%title('Frequency of word $j$ in nonspam class');
%xlabel('Word label'); ylabel('Empirical Frequency');
%subplot(2,1,2);
%line([1:d;1:d],[zeros(size(1:d));theta(:,2)'],'color','k','linewidth',3)
%title('Frequency of word $j$ in spam class');
%xlabel('Word label'); ylabel('Empirical Frequency')

% For color, this is better
figure(); hold on;
nonspam = line([(1:d)-1/2;(1:d)-1/2],[zeros(size(1:d));theta(:,1)'],'color','k','linewidth',3);
spam = line([(1:d);(1:d)],[zeros(size(1:d));theta(:,2)'],'color','r','linewidth',3);
title('Frequency of features in spam and nonspam email');
legend([nonspam(1),spam(1)],'nonspam', 'spam','location','best');
xlabel('Feature label'); ylabel('Empirical Frequency');
pdfcrop; print_pdf('featurefreq-class');

% Plot the mutual information for the features minus the last three which are uninformative
figure(); hold on;
line([1:54;1:54],[zeros(1,54);mi(1:54)],'color','k','linewidth',3);
title('Mutual information for features (excluding uninformative features)');
xlabel('Feature label'); ylabel('Mutual Information');
pdfcrop; print_pdf('featuremi');



% Our custom score function - to ensure consistency across all methods
function [score,stdErr] = scoreFunction(obj,model)

            n = size(obj.Xdata,1);
            scoreArray = zeros(n,1);
            for f = 1:obj.CVnfolds
                myXtrain = obj.Xdata(trainfolds{f},:);
                myytrain = obj.Ydata(trainfolds{f},:);
                myXtest  = obj.Xdata(testfolds{f} ,:);
                myytest  = obj.Ydata(testfolds{f},:);
                scoreArray(testfolds{f}) = obj.lossFunction(obj.predictFunction(myXtrain,myytrain,myXtest,model{:}),myXtest,myytest);
            end
            score = mean(scoreArray);
            stdErr = std (scoreArray)/sqrt(n);
end

end
