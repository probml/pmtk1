function [precMat, covMat, klDiv] = gaussIPF(X, G)

covMat = cov(X);
[precMat, klDiv] = helper(G, covMat);
covMat = inv(precMat);

%%%%%%%

function [J klDiv] = helper(adjacency, P, dim, tol, maxIts)
%gaussIPF    Gaussian iterative proportional fitting (IPF).
%   Iterative proportional fitting algorithm to approximate a given 
%   covariance with a Gaussian MRF.  Represents current model by inverse
%   covariance so that appropriate sparsity guaranteed.
%
%     [J, klDiv] = gaussIPF(adjacency, P, dim, tol, maxIts)
%
% PARAMETERS:
%   adjacency = square adjacency matrix (assumed symmetric) which has a
%     row/column for each node.  Edges are determined by nonzero entries.
%   P = target covariance matrix (only need entries for edges)
%   dim = scalar giving dimension of hidden variable at each node (DEFAULT = 1)
%   tol = convergence tolerance (DEFAULT = 1e-8)
%   maxIts = maximum number of iterations (DEFAULT = 100)
% OUTPUT:
%   J = sparse inverse covariance approximation with structure of adjacency
%   klDiv = KL divergence of approximation after each IPF iteration

% Erik Sudderth
%  February 19, 2003 - Initial version completed


% Check parameters
if (nargin < 2)
  error('Invalid number of parameters.');
end
if (nargin < 3)
  dim = 1;
end
if (nargin < 4)
  tol = 1e-8;
end
if (nargin < 5)
  maxIts = 100;
end

% Verify that adjacency matrix and target covariance are consistent
N = length(adjacency);
if (N*dim ~= length(P))

  error('Parameter size mismatch');
end

% Extract edge information from adjacency matrix
[adjRow, adjCol] = find(adjacency);
I = find(adjRow > adjCol);
numEdges = length(I);
edgeRow = adjRow(I);
edgeCol = adjCol(I);

% Initialize variables
J = sparse([1:N*dim],[1:N*dim],ones(1,N*dim),N*dim,N*dim,(2*numEdges+N)*dim^2);

if (det(P) < tol)
  klDiv = 1/tol;
else
  klDiv = -0.5*(log(det(P)) + trace(eye(N*dim)-P));
end
%fprintf(1,'klDiv = %g\n', klDiv);

% Iterate until change in KL divergence falls below tol
for (s = 2:maxIts)
  if (s >= 3 & abs(klDiv(s-1) - klDiv(s-2)) < tol)
    break;
  end

  for (t = 1:numEdges)
    C = [(edgeRow(t)-1)*dim+1:edgeRow(t)*dim, ...
         (edgeCol(t)-1)*dim+1:edgeCol(t)*dim];

    y = sparse(C,[1:2*dim],ones(1,2*dim),N*dim,2*dim);
    PmodelCols = J\y;
    Jmodel = inv(PmodelCols(C,:)); 

    Jtgt = inv(P(C, C));
    J(C,C) = J(C,C) + Jtgt - Jmodel;
  end

  klDiv(s) = -0.5*(log(det(P*J)) + trace(eye(N*dim)-P*J));
% fprintf(1,'klDiv = %g\n', klDiv(s));
end
