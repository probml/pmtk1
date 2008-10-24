function [mseTest, w]= prostateSubsets(X, y, ndxTrain, ndxTest)

% Reproduce fig 3.6 on p58 of "Elements of statistical learning" 

Nfolds = 10;
xtrainAll = X(ndxTrain,:); ytrainAll = y(ndxTrain);
seed= 0; rand('state', seed);
[n d] = size(xtrainAll);
sizes = 0:d;
perm = randperm(n);
xtrainAll = xtrainAll(perm,:);
ytrainAll = ytrainAll(perm);
[trainfolds, testfolds] = Kfold(size(xtrainAll,1), Nfolds);
clear mseCVTrain mseCVTest
for f=1:length(trainfolds)
    xtrain = xtrainAll(trainfolds{f},:);
    ytrain = ytrainAll(trainfolds{f});
    xtestCV = xtrainAll(testfolds{f},:);
    ytestCV = ytrainAll(testfolds{f});
    [w, mseCVTrainAll(f,:), mseCVTestAll(f,:), sz, members, df, mseCVTest(f,:)] = ...
	allSubsetsRegression(xtrain, ytrain, xtestCV, ytestCV, sizes, 1);
end
mseCVMean = mean(mseCVTest,1);
mseCVse = std(mseCVTest,[],1)/sqrt(Nfolds);

figure;
errorbar(df, mseCVMean, mseCVse);
kstar = oneStdErrorRule(mseCVMean, mseCVse);
hold on
ax = axis;
line([df(kstar) df(kstar)], [ax(3) ax(4)], 'Color', 'r', 'LineStyle', '-.');
xlabel('size of subset')
ylabel('cv error')


% Refit using all training data to find set of chosen size
[w, mseTrain, junk, junk2, members] = ...
    allSubsetsRegression(X(ndxTrain,:), y(ndxTrain), [], [], sizes(kstar), 1);
[junk,best]=min(mseTrain);
ndx = members{best};
[ww, mseTrain, mseTest]  = ridgeQR(X(ndxTrain,ndx), y(ndxTrain), ...
				  X(ndxTest,ndx), y(ndxTest), 0, 1);
w = zeros(d,1);
w(ndx) = ww(2:end);
w = [ww(1); w];

title(sprintf('all subsets, mseTest = %5.3f', mseTest))
