function [w] = lassoExp(X, y, t,varargin)
% Naive implementation of Lasso
% Based on code by Mark Schmidt

[maxIter,verbose,display] = process_options(varargin,'maxIter',10000,'verbose','1','Display','none');
[n p] = size(X);
warning off all;

% Create Exponential number of sign constraints
a=[0:2^p-1]';
for i=1:p,
    A(:,i)=2*bitand(a,1)-1;
    a=bitshift(a,-1);
end
b = t * ones(2^p,1);

% Start from the Least Squares solution (optional)
w = pinv(X'*X)*(X'*y); %w = minL2(X,y);

% Run the QP (bounding below seems to speed convergence)
options = optimset('Display',display,'MaxIter',maxIter);
[w fval exitflag output lambda] = quadprog(X'*X,-y'*X,A,b,[],[],-t*ones(p,1),t*ones(p,1),w,options);
