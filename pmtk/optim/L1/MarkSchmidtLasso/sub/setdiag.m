function M = setdiag(M, v)
% SETDIAG Set the diagonal of a matrix to a specified scalar/vector.
% M = setdiag(M, v)
% e.g., for 3x3 matrix,  elements are numbered
% 1 4 7 
% 2 5 8 
% 3 6 9
% so diagnoal = [1 5 9]

n = size(M,1);
M(1:n+1:n^2) = v;

