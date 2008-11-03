function precMat = covselProj(C, G, useC)
% Find MLE precision matrix given covariance matrix C and GGM graph G
% aka covariance selection
% Uses the L1 projection method described in
% Projected Subgradient Methods for Learning Sparse Gaussian
% Duchi, Gould, Koller UAI'08

%# author Ewout Vandenbder vandenberg.ewout@gmail.com

if nargin < 3, useC = false; end
d = size(C,1);
groups = reshape(1:(d^2),d,d);
GG = setdiag(G,1); % only structural zeros left
Lambda = 1e5*(1-GG); % 0 edges get penalized
precMat = Algorithm3(C,groups,Lambda, useC);
end

function [K,W] = Algorithm3(Sigma, groups, lambda, useC)

% Get problem size and number of groups
n = size(Sigma,1);
nGroups = max(max(groups));

% Find initial W, using lemma 1 and diag(W) = lambda
W = initialW(Sigma,lambda(diag(groups)));
%W = projectLinf1(W,lambda);
K = inv(Sigma + W);

% Print header
%fprintf('%4s  %11s %9s %9s\n','Iter','Objective','Gap','Step');

% Main loop
i = 0;  maxiter = 100; epsilon = 1e-3; alpha = 1e-2; beta = 0.5; t = 1;
while (1)
   % Compute unconstrained gradient
   G = K;

   % Compute direction of step
   D = projectLinf1(W+t*G,groups,lambda, useC) - W;

   % Compute initial and next objective values
   f0 = logdet(Sigma + W,-Inf);
   ft = logdet(Sigma + W + D,-Inf);

   % Perform backtracking line search
   while ft < f0 + alpha * traceMatProd(D,G)
      % Exit with alpha is too small
      if (t < 1e-6), break; end;

      % Decrease t, recalculate direction and objective
      t = beta * t;
      D = projectLinf1(W + t*G,groups,lambda, useC) - W;
      ft = logdet(Sigma + W + D,-Inf);
   end
   f = ft;

   % Update W and K
   W = W + D;
   K = inv(Sigma + W);

   % Compute duality gap
   eta = traceMatProd(Sigma,K) - n;
   for j=1:nGroups
      mak = max(abs(K(groups == j)));
      if ~isempty(mak)
         eta = eta + lambda(j) * mak;
      end;
   end

   % Increment iteration
   i = i + 1;

   % Print progress
   %fprintf('%4d  %11.4e %9.2e %9.2e\n',i,f,eta,t);

   % Increase t slightly
   t = t / beta;

   % Check stopping criterion
   if (eta < epsilon)
      fprintf('Exit: Optimal solution\n');
      break;
   elseif (i >= maxiter)
      fprintf('Exit: Maximum number of iterations reached\n');
      break;
   elseif (t < 1e-6)
      fprintf('Exit: Linesearch error\n');
      break;
   end
end
end

function W = initialW(Sigma,lambda)
alpha = 1;
W     = (alpha - 1) * Sigma + (1-alpha) * diag(diag(Sigma)) + diag(lambda);
end

function M = projectLinf1(M,groups,lambda, useC)
% Just testing: Each element in its own group :p
%lambda = lambda(groups);
%idx = M > lambda; M(idx) =  lambda(idx); % Project - Positive part
%idx = M <-lambda; M(idx) = -lambda(idx); % Project - Negative part
%return

nGroups = max(max(groups));
for i=1:nGroups
   idx = groups == i;
   x = M(idx);
   if isempty(x), continue; end;

   M(idx) = projectL1(x,lambda(i), useC);
end
end

function p = projectL1(p,tau, useC)
s = sign(p);
if useC
  p = projectRandom2C(abs(p),tau);
else
  p = projectRandom2(abs(p),tau);
end
p = p.*s;
end

function t = traceMatProd(A,B)
% Compute trace(A*B)
t = sum(sum(A.*B'));
end

function l = logdet(M,errorDet)
[R,p] = chol(M);
if p ~= 0
   l = errorDet;
else
   l = 2*sum(log(diag(R)));
end;
end

