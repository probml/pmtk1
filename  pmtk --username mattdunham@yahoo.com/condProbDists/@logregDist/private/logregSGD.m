function [w, output] = logregSGD(X, y, lambda, varargin)
% X(i,:) is i'th case
% y(i) = 0 or 1
% lambda l2 regularizer
% optional arguments as in minFunc - MaxIter and TolX
%
% output currently only contains ftrace

[maxIter, tolX, eta] = process_options(varargin, ...
  'maxIter', 50, 'tolX', 1e-4);
check01(y);
[n d] = size(X);
w = rand(d,1);
wold = w;
ftrace = [];
eta = 1;
for iter=1:maxIter
  for i=1:n
    xi = X(i,:)';
    mui = sigmoid(w'*xi);
    gi = (y(i)-mui)*xi - (lambda/n)*w;
    w = w + eta*gi;
    incr = norm(wold -w)/norm(wold);
    if (incr < tolX)
      break;
    end
    wold = w;
  end
  eta = 1/(iter+1) % step size decay
  if nargout > 1
    f = logregNLLgradHess(w, X, y, lambda);
    ftrace(iter) = f;
  end
end
output.ftrace = ftrace;
end

   