function [w,b] = perceptron(X, y,lambda)
% X(i,:) is i'th case, y(i) = -1 or +1
% X should not contain a column of 1s
% lambda l2 regularizer
% Based on code by Thomas Hoffman

    [n d] = size(X);
    w = zeros(d,1);
    b = zeros(1,1); % offset term
    max_iter = 100000;
    for iter=1:max_iter
        errors = 0;
        for i=1:n
            xi = X(i,:)';
            yhati = sign(w'*xi + b);
            if ( y(i)*yhati <= 0 ) % made an error
                w = w + y(i) * xi - (lambda/n)*w;
                b = b + y(i);
                errors = errors + 1;
            end
        end
        %fprintf('Iteration %d, errors = %d\n', iter, errors);
        if (errors==0)
            break;
        end
    end
