clear all
close all

load('prostate.mat') % from prostateDataMake
[n d] = size(X);
ndxTrain = find(istrain);
ndxTest = setdiff(1:n, ndxTrain);


[mseTestLS, wLS] = prostateLS(X, y, ndxTrain, ndxTest);
[mseTestRidge, wRidge] = prostateRidge(X, y, ndxTrain, ndxTest);;
[mseTestSS, wSS] = prostateSubsets(X, y, ndxTrain, ndxTest);;
[mseTestLasso, wLasso] = prostateLasso(X, y, ndxTrain, ndxTest);;


fprintf('%10s %7s %7s %7s %7s\n',...
	'Term', 'LS', 'Subset', 'Ridge', 'Lasso');
fprintf('%10s %7.3f %7.3f %7.3f %7.3f\n',...
	'intercept', wLS(1), wSS(1), wRidge(1), wLasso(1));
for i=1:d
  fprintf('%10s %7.3f %7.3f %7.3f %7.3f\n',...
	  names{i}, wLS(i+1), wSS(i+1), wRidge(i+1), wLasso(i+1));
end  
fprintf('\n%10s %7.3f %7.3f %7.3f %7.3f\n',...
	'Test MSE', mseTestLS, mseTestSS, mseTestRidge, mseTestLasso);


