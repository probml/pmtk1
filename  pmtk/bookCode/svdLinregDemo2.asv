

X = reshape(1:15, [3 5])';
wtrue = [-15.333, 15.667, 0]';
y = X*wtrue;

[U,S,V]=svd(X,'econ')
w = V*inv(S)*U'*y
%w = V*pinv(S)*U'*y

s = diag(S);
ndx=find(abs(s)<1e-10);
z = V(:,ndx) %  [0.41 -0.82 0.41], mkUnitNorm([1,-2,1])

% add in any multiple of z to the data to determine unique solution
xnew = z;
ynew = xnew'*wtrue;
X = [X; xnew']
y = [y; ynew]

[U,S,V]=svd(X,'econ');
w = V*inv(S)*U'*y
assert(approxeq(w,wtrue))