function lambdaVals = recoverLambda(X,y,Wfull)
% Recover the l1 regularization constants that would result in the
% regression weights in Wfull were lasso to be performed on (X,y). Sets
% of regression weights, (one per lambda) are stacked row-wise in
% Wfull, as returned by lars.
W = Wfull';
lambdaVals = 2*max(abs(X'*(bsxfun(@minus,y,X*W))),[],1);
end