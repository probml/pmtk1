function [f,g,H] = multinomLogregNLLGradHessL2(w, X, Y, lambda,offset)
% Return the negative log likelihood for multinomial logistic regression
% with an L2 regularizer, lambda. Also return the gradient and Hessian of
% the penalized nll. This function is designed to be passed to fminunc to
% minimize the penalized nll = f(w,lambda). 
%
% Inputs:
% w             ndimensions*(nclasses-1)-by-1
% X             nexamples-by-ndimensions
% Y             nexamples-by-nclasses (1 of C encoding)
% lambda        L2 regularizer ndimensions*1
% offset        if true, (i.e. column of ones added to left of X), offset weights are
%               not penalized.
%
% Outputs:
% f             L2 penalized nll
% g             gradient of f
% H             Hessian of f
% Note, here we minimize the nll rather than maximize the ll as done in the SMLR
% paper, hence the sign change. 

%#author Balaji Krishnapuram
%#modified Kevin Murphy, Matt Dunham

if(nargin <5)
    offset = false;
end

[nexamples,ndimensions] = size(X);
nclasses = size(Y,2); 
P=(multiSigmoid(X,w)); % P->(nclasses-by-nexamples)

if(offset)
   w = reshape(w,ndimensions,nclasses-1); 
   w(1,:) = 0; % don't penalize offset, (note P already calculated with original w)
   w = w(:);
end

%% NLL
f =  sum(sum(Y.*log(P)));                       % log likelihood
f = f - (lambda/2).*w'*w;                       % penalized log likelihood
f = -f;                                         % penalized negative log likelihood
%f = f./nexamples;                              % scale objective function by n
if nargout < 2, return; end
%% Gradient
dims = 1:(nclasses-1);                          % gradient of log likelihood
g = X'*( Y(:,dims)-P(:,dims) );                 % gradient of penalized log likelihood
g = g(:) - lambda.*w;                           % gradient of penalized negative log likelihood
g = -g;                   
%g = g./nexamples;
if nargout < 3, return; end
%% Hessian
H = zeros(ndimensions*(nclasses-1), ndimensions*(nclasses-1));
% eq between 7,8 in SMLR paper
for i=1:nexamples
  xi = X(i,:)';
  mui = P(i,1:nclasses-1)';
  A = diag(mui) - mui*mui';
  H = H - kron(A,xi*xi');
end                                             % Hessian of log likelihood
H = H - lambda.*eye(size(H));                   % Hessian of penalized log likelihood
H = -H;                                         % Hessian of penalized negative log likelihood
%H = H./nexamples;
%% Sanity Check
if(0)

% Equation 9, SMLR paper is equivalent to our vectorized version
gTest=zeros((nclasses-1)*ndimensions,1);          
for i=1:nexamples
  gTest=gTest+kron((Y(i,1:(nclasses-1))-P(i,1:(nclasses-1)))',(X(i,:))');
end
gTest = gTest - lambda*w(:);
gTest = -gTest;
gTest = gTest./nexamples;
assert(approxeq(g, gTest))

end
