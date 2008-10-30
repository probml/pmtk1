function [precMat, covMat] = glassoR(X, varargin)
% Use R code to find L1-penalized precision matrix
% R code from http://www-stat.stanford.edu/~tibs/glasso/
% For instructions on calling R from Matlab, see
% http://www.cs.ubc.ca/~mdunham/tutorial/external.html#21

[rho, useMBapprox, junk] = process_options(...
    varargin, 'regularizer', [], 'useMB', 0);

[precMat, covMat] = helper(cov(X), rho, useMBapprox);
if ~useMBapprox
  assert(isposdef(precMat))
  assert(isposdef(covMat))
end

%%%%%%%%

function [precMat, covMat] = helper(C, rho, useMBapprox)


if 0
  x <- matrix(rnorm(50*20),ncol=20)
  s <- var(x)
  a <- glasso(s, rho=0.1)
end

openR;
evalR('C<-1') % must pre-declare variable before writing a matrix
evalR('L<-1') 
putRdata('C',C);
putRdata('rho',rho);
putRdata('useMBapprox', useMBapprox)
evalR('stuff <- glasso(C,rho=rho,approx=useMBapprox)'); 
evalR('L <- stuff$wi') % inverse covariance matrix
precMat = getRdata('L');
evalR('L <- stuff$w') %  covariance matrix
covMat = getRdata('L');
closeR;
