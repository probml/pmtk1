function covshrinkKPMdemo
% Based on testscript-matlab.txt from http://strimmerlab.org/software.html

load smalldata.txt   % n=6,p=10
X=smalldata; 
demo(X)

load largedata.txt  % n=20,p=100
X=largedata;
demo(X)

%n=6, p=10
%shrinkage correlation 0.7315, variance 0.6015
%rank MLE = 5.0000, shrunk = 10.0000
%psd  MLE = 0, shrunk = 1
%n=20, p=100
%shrinkage correlation 0.8851, variance 0.7772
%rank MLE = 19.0000, shrunk = 100.0000
%psd  MLE = 0, shrunk = 1

%%%%%%%

function demo(X)

fprintf('n=%d, p=%d\n', size(X,1), size(X,2));
s1 = cov(X);     
[s2, lamcor, lamvar] = covshrinkKPM(X, 1);    
fprintf('shrinkage correlation %5.4f, variance %5.4f\n', lamcor, lamvar);
fprintf('rank MLE = %5.4f, shrunk = %5.4f\n', rank(s1), rank(s2)); 
fprintf('psd  MLE = %d, shrunk = %d\n', all(eig(s1)>0), all(eig(s2)>0));
