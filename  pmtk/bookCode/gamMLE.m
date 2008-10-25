function [a,b] = gamMLE(X)
% MLE for Gamma a = shape, b= rate (not scale)
params = gamfit(X);
a = params(1);
b = 1/params(2);
