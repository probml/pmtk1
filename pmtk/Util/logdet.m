function y = logdet(A)
% log(det(A)) where A is positive-definite.
% This is faster and more stable than using log(det(A)).

%#author Tom Minka
%#url http://research.microsoft.com/~minka/software/lightspeed/

% Written by Tom Minka
% (c) Microsoft Corporation. All rights reserved.

[U,p] = chol(A);
posdef = (p == 0);
if ~posdef
  y = NaN;
else
  y = 2*sum(log(diag(U)));
end