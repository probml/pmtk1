function [w] = perceptron(X, y,lambda)
% X(i,:) is i'th case, y(i) = -1 or +1
% lambda l2 regularizer
% Based on code by Thomas Hoffman

[n d] = size(X);
w = zeros(d,1);
max_iter = 1000;
for iter=1:max_iter
  errors = 0;
  for i=1:n
    xi = X(i,:)';
    yhati = sign(w'*xi);
    if ( y(i)*yhati <= 0 ) % made an error
      w = w + y(i) * xi - (lambda/n)*w;
      errors = errors + 1;
    end
  end
  %fprintf('Iteration %d, errors = %d\n', iter, errors);
  if (errors==0)
    break;
  end
end
