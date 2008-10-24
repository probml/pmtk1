function K = makeKernelMatrix(X1, X2, kernel)
% K(i,j) = k(X1(i,:), X2(j,:), params)

switch kernel.type
 case 'rbf'
  K = rbfKernel(X1, X2, kernel.sigma);
 case 'poly',
  K = polyKernel(X1, X2, kernel.deg, kernel.b);
 otherwise
  error(['unknown kernel type ' kernel.type])
end

function K = polyKernel(X1, X2, deg, b)
% K(i,j) = (<X1(i,:), X2(j,:)> + b)^deg, where b defaults to 0

if nargin < 4, b = 0; end
K = (X1*X2' + b).^deg;

function K = rbfKernel(X1, X2, sigma)
% K(i,j) = exp(-1/(2*sigma^2)  ||X1(i,:) - X2(j,:)||^2 )

S = sqdist(X1',X2');
%K = exp(-1/(2*sigma^2) * S);
K = exp(-(1/sigma^2) * S);
