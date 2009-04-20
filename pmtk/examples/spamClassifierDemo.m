function [model,baseModel] = spamClassifierDemo()
%% Spam Classifier Demo
% Perform classification on spam demo, using dataset from Hastie, Tribshirani,and Friedman
%#author Cody Severinski

doprint = false;    
    
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
nModels = 8;

% lambda for l2-regularization
nLambda = 10;
lambda = logspace(-10,-1,nLambda);

% Get indices for each of the folds
[trainfolds,testfolds] = Kfold(n,K);

% A function handle to our custom score function.  This is needed to ensure that we use the same folds regardless of whether we use the ModelSelection class or not
cvCustom = @(obj,model)scoreFunction(obj,model);

modelName = {'Naive Bayes', 'Logistic Regression', 'Diagonal Gaussian', 'Logistic Regression', 'Laplace Approximation', 'Laplace Approximation', 'Naives Bayes', 'Logistic Regression'};
dataName = {'Binary', 'Binary', 'log-transformed', 'log-transformed', 'continuous', 'log-transformed', 'Binary (last three omitted)', 'Binary (last three omitted)'};

midx = 1;
fprintf('Constructing Base models: %s, %s \n', modelName{midx}, dataName{midx});
% Naive Bayes
% Define discrete class conditionals, with support on [0,1], taking on one of two classes; Define equal probability priors
baseModel{midx}.classConditionals = copy(DiscreteDist('-support',[0,1]),1,2);
baseModel{midx}.classPrior = DiscreteDist('-T',normalize(ones(2,1)),'-support',0:1);
baseModel{midx}.model = GenerativeClassifierDist('classConditionals',baseModel{midx}.classConditionals,'classPrior',baseModel{midx}.classPrior);
baseModel{midx}.X = binspam;

midx = 2;
fprintf('Constructing Base models: %s, %s \n', modelName{midx}, dataName{midx});
% Logistic regression on binary data
baseModel{midx}.model = @(lambda)LogregDist('prior','l2','priorStrength',lambda);
baseModel{midx}.modelspace = ModelSelection.makeModelSpace(lambda);
baseModel{midx}.predictFunction = @(Xtrain,ytrain,Xtest,lambda)...
	mode(predict(fit(baseModel{midx}.model(lambda),'X',Xtrain,'y',ytrain),Xtest));
baseModel{midx}.X = binspam;
baseModel{midx}.modelselection = ModelSelection(              ...
		  'predictFunction',baseModel{midx}.predictFunction ,...      % the test function we just created
			'scoreFunction',cvCustom ,...
		  'Xdata'       ,baseModel{midx}.X         ,...      % all of the X data we have available
		  'Ydata'       ,classLabel             ,...      % all of the y data we have available
		  'verbose'     ,true            ,...      % turn off progress report
		  'models'      ,baseModel{midx}.modelspace					,...% the model space created above
			'CVnfolds'		, 5 );
baseModel{midx}.lambda = baseModel{midx}.modelselection.bestModel{1};
baseModel{midx}.model = baseModel{midx}.model(baseModel{midx}.lambda);

midx = 3;
fprintf('Constructing Base models: %s, %s \n', modelName{midx}, dataName{midx});
% Diagonal Gaussian on log transformed
% Define discrete class conditionals, with support on [0,1], taking on one of two classes
baseModel{midx}.classConditionals = copy(MvnDist(zeros(1,d),diag(1*ones(1,d)),'-prior','nig','-covtype','diagonal'),1,2);
%baseModel{midx}.classConditionals = copy(MvnDist(zeros(1,d),diag(1*ones(1,d)),'-covtype','diagonal'),1,2);
% Define equal probability priors
baseModel{midx}.classPrior = DiscreteDist('-T',normalize(ones(2,1)),'-support',0:1);
baseModel{midx}.model = GenerativeClassifierDist('classConditionals',baseModel{midx}.classConditionals,'classPrior',baseModel{midx}.classPrior);
baseModel{midx}.X = logspam;

midx = 4;
fprintf('Constructing Base models: %s, %s \n', modelName{midx}, dataName{midx});
% logistic regression using log transformed
baseModel{midx}.model = @(lambda)LogregDist('prior','l2','priorStrength',lambda);
baseModel{midx}.modelspace = ModelSelection.makeModelSpace(lambda);
baseModel{midx}.predictFunction = @(Xtrain,ytrain,Xtest,lambda)...
   mode(predict(fit(baseModel{4}.model(lambda),'X',Xtrain,'y',ytrain),Xtest));
baseModel{midx}.X = logspam;
baseModel{midx}.modelselection = ModelSelection(              ...
		  'predictFunction',baseModel{midx}.predictFunction ,...      % the test function we just created
			'scoreFunction',cvCustom ,...
		  'Xdata'       ,baseModel{midx}.X          ,...      % all of the X data we have available
		  'Ydata'       ,classLabel             ,...      % all of the y data we have available
		  'verbose'     ,true            ,...      % turn off progress report
		  'models'      ,baseModel{midx}.modelspace ,...
			'CVnfolds'		, 5					);        % the model space created above
baseModel{midx}.lambda = baseModel{midx}.modelselection.bestModel{1};
baseModel{midx}.model = baseModel{midx}.model(baseModel{midx}.lambda);

midx = 5;
fprintf('Constructing Base models: %s, %s \n', modelName{midx}, dataName{midx});
% now the laplace approximation to the continuous data
baseModel{midx}.model = @(lambda)Logreg_MvnDist('-infMethod','laplace','-priorStrength',lambda);
baseModel{midx}.modelspace = ModelSelection.makeModelSpace(lambda);
baseModel{midx}.X = spam;
baseModel{midx}.predictFunction = @(Xtrain,ytrain,Xtest,lambda)...
   mode(predict(fit(baseModel{midx}.model(lambda),'X',Xtrain,'y',ytrain),Xtest));
baseModel{5}.modelselection = ModelSelection(              ...
		  'predictFunction',baseModel{midx}.predictFunction ,...      % the test function we just created
			'scoreFunction',cvCustom ,...
		  'Xdata'       ,baseModel{midx}.X          ,...      % all of the X data we have available
		  'Ydata'       ,classLabel             ,...      % all of the y data we have available
		  'verbose'     ,true            ,...      % turn off progress report
		  'models'      ,baseModel{midx}.modelspace     ,...% the model space created above
			'CVnfolds'		, 5);        
baseModel{midx}.lambda = baseModel{midx}.modelselection.bestModel{1};
baseModel{midx}.model = baseModel{midx}.model(baseModel{midx}.lambda);

midx = 6;
fprintf('Constructing Base models: %s, %s \n', modelName{midx}, dataName{midx});
% now the laplace approximation to the continuous data
baseModel{midx}.model = @(lambda)Logreg_MvnDist('-infMethod','laplace','-priorStrength',lambda);
baseModel{midx}.modelspace = ModelSelection.makeModelSpace(lambda);
baseModel{midx}.X = logspam;
baseModel{midx}.predictFunction = @(Xtrain,ytrain,Xtest,lambda)...
   mode(predict(fit(baseModel{midx}.model(lambda),'X',Xtrain,'y',ytrain),Xtest));
baseModel{midx}.modelselection = ModelSelection(              ...
		  'predictFunction',baseModel{midx}.predictFunction ,...      % the test function we just created
			'scoreFunction',cvCustom ,...
		  'Xdata'       ,baseModel{midx}.X          ,...      % all of the X data we have available
		  'Ydata'       ,classLabel             ,...      % all of the y data we have available
		  'verbose'     ,true            ,...      % turn off progress report
		  'models'      ,baseModel{midx}.modelspace     ,...% the model space created above
			'CVnfolds'		, 5);        
baseModel{midx}.lambda = baseModel{midx}.modelselection.bestModel{1};
baseModel{midx}.model = baseModel{midx}.model(baseModel{midx}.lambda);

midx = 7;
fprintf('Constructing Base models: %s, %s \n', modelName{midx}, dataName{midx});
% Naive Bayes without the last three uninformative feature
% Define discrete class conditionals, with support on [0,1], taking on one of two classes; Define equal probability priors
baseModel{midx}.classConditionals = copy(MvnDist(zeros(1,d),diag(1*ones(1,d)),'-prior','nig','-covtype','diagonal'),1,2);
%baseModel{midx}.classConditionals = copy(DiscreteDist('support',[0,1]),1,2);
baseModel{midx}.classPrior = DiscreteDist('-T',normalize(ones(2,1)),'-support',0:1);
baseModel{midx}.model = GenerativeClassifierDist('classConditionals',baseModel{midx}.classConditionals,'classPrior',baseModel{midx}.classPrior);
baseModel{midx}.X = binspam(:,1:54);

midx = 8;
fprintf('Constructing Base models: %s, %s \n', modelName{midx}, dataName{midx});
% Logistic regression on binary data
baseModel{midx}.model = @(lambda)LogregDist('prior','l2','priorStrength',lambda);
baseModel{midx}.modelspace = ModelSelection.makeModelSpace(lambda);
baseModel{midx}.predictFunction = @(Xtrain,ytrain,Xtest,lambda)...
	mode(predict(fit(baseModel{midx}.model(lambda),'X',Xtrain,'y',ytrain),Xtest));
baseModel{midx}.X = binspam(:,1:54);
baseModel{midx}.modelselection = ModelSelection(              ...
		  'predictFunction',baseModel{midx}.predictFunction ,...      % the test function we just created
			'scoreFunction',cvCustom ,...
		  'Xdata'       ,baseModel{midx}.X         ,...      % all of the X data we have available
		  'Ydata'       ,classLabel             ,...      % all of the y data we have available
		  'verbose'     ,true            ,...      % turn off progress report
		  'models'      ,baseModel{midx}.modelspace					,...% the model space created above
			'CVnfolds'		, 5 );
baseModel{midx}.lambda = baseModel{midx}.modelselection.bestModel{1};
baseModel{midx}.model = baseModel{midx}.model(baseModel{midx}.lambda);


for mod=1:nModels
	fprintf('Fitting: %s, %s \n', modelName{mod}, dataName{mod});
	% Fit the naive bayes classifier
    
	for fold=1:K
		model{mod,fold}.classifier	= fit( baseModel{mod}.model,'X',baseModel{mod}.X(trainfolds{fold},:),'y',classLabel(trainfolds{fold}) );
		model{mod,fold}.predict			= predict( model{mod,fold}.classifier, baseModel{mod}.X(testfolds{fold},:) );
		model{mod,fold}.yhat				= mode( model{mod,fold}.predict );
		model{mod,fold}.err					= mean( model{mod,fold}.yhat ~= classLabel(testfolds{fold}) );
		%[model{mod,fold}.FPrate,model{mod,fold}.TPrate,model{mod,fold}.auc, model{mod,fold}.threshold] = computeROC( model{mod,fold}.predict.mu(2,:),classLabel(testfolds{fold}) );
        pmat = pmf(model{mod,fold}.predict);
        [model{mod,fold}.FPrate,model{mod,fold}.TPrate,model{mod,fold}.auc, model{mod,fold}.threshold] = computeROC( pmat(2,:),classLabel(testfolds{fold}) );
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
plotParam = {'b-','g-','r-','c-','m-','y-','bx:','gx:'};
plotMarker = {'b', 'g', 'r', 'c', 'm', 'y', 'k', 'k'};
plotMarkerSize = {1, 1, 1, 1, 1, 1, 5, 5};
figure(); hold on;
for mod=1:nModels
foldidx = argmax(auc(mod,:));
plot(model{mod,foldidx}.FPrate, model{mod,foldidx}.TPrate,plotParam{mod},'MarkerFaceColor',plotMarker{mod},'linewidth',3);
xlabel('false positive rate'); ylabel('true positive rate'); %title(sprintf('ROC Curve: %s (%s Data)',modelName{mod},dataName{mod}));
title('ROC Curves for different spam classifiers');
end
line([0,1],[1,0],'color','k','linewidth',3);
legendnames = cell(1,nModels);
for i=1:nModels
	legendnames{i} = strcat(modelName{i}, ', ', dataName{i});
end
legend(legendnames,'location','best');
axis([0, 0.40, 0.60, 1]);
if(doprint)
    pdfcrop; print_pdf('rocCurves');
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
if(doprint)
    pdfcrop; print_pdf(sprintf('laplaceapprox-marginal-%s',num2str(mod)));
end
end







% Now compute mutual information
theta = [mean(binspam(classLabel == 0,:))', mean(binspam(classLabel == 1,:))'];
[mi] = NBmi(theta);
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
if(doprint)
    pdfcrop; print_pdf('featurefreq-class');
end
% Plot the mutual information for the features minus the last three which are uninformative
figure(); hold on;
line([1:54;1:54],[zeros(1,54);mi(1:54)],'color','k','linewidth',3);
title('Mutual information for features (excluding uninformative features)');
xlabel('Feature label'); ylabel('Mutual Information');
if(doprint)
    pdfcrop; print_pdf('featuremi');
end



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
