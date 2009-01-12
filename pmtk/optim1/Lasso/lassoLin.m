function [w,exitflag] = lassoLin(X, y, t,varargin)
% Lasso using positive variables,

%#url http://www.cs.ubc.ca/~schmidtm/Software/lasso.html
%#author Mark Schmidt
%#modified  Kevin Murphy

[maxIter,verbose,display,optTol] = process_options(varargin,'maxIter',10000,...
   'verbose','1','Display','none','optTol',0.0000001);
[n p] = size(X);
warning off all

% Use modified Design matrix and create constraints
X = [X -X];
A = [-1*eye(2*p,2*p);ones(1,2*p)];
b = [zeros(2*p,1);t];

% Run the QP
options = optimset('Display',display,'MaxIter',maxIter);
[w2] = quadprog(X'*X,-y'*X,A,b,[],[],-t*ones(2*p,1),t*ones(2*p,1),zeros(2*p,1),options);
w = [w2(1:p)-w2(p+1:2*p)];
