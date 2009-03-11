function [precMat, covMat] = ggmLassoCoordDesc(S, lambda, varargin)
% L1 regularized MLE on precision matrix using  graohical lasso
% C = cov(data)
% Uses algorithm "Sparse inverse covariance estimation the
% graphical lasso", Friedman 2007
% See "Elements of statistical learning" 2ed p636

error('not finished')
optTol = 0.00001;
p = size(S,1);
maxIter = 20;
W = S + lambda*eye(p,p);

iter = 1;
done = false;
while ~done
  for i = 1:p
    noti = setdiffPMTK(1:p, i);
    s_12 = S(noti,i);
    W_11 = W(noti,noti);
    w = shooting(W_11, s_12, lambda); % ** implement this function
    W(noti,i) = w;
    W(i,noti) = w';
  end
  iter = iter + 1;
  converged = X; % ** implement convergence check
  done = converged | (iter > maxIter);
end

covMat = W;
%precMat = inv(W); % ** implement more efficient method (step 3 of alg 17.2)
end
