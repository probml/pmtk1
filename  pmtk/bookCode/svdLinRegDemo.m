% Beers  p147

wtrue = [1 2 3]';
X = [ones(4,1) (0:3)' (0:3)']
y = X*wtrue

[U,S,V]=svd(X,'econ')
%b = V*inv(S)*U'*y
b = V*pinv(S)*U'*y

% add in any multiple of V(:,3) to the data
%xnew = X(3,:) + (1/0.7071)*V(:,3)';
%xnew = X(3,:) + 1*V(:,3)';
xnew = V(:,3)';
ynew = xnew*wtrue;
X = [X; xnew]
y = [y; ynew]

[U,S,V]=svd(X,'econ');
b = V*inv(S)*U'*y
assert(approxeq(b,wtrue))
