function precMat = covselPython(C, G)
% Find MLE precision matrix given covariance matrix C and GGM graph G
% aka covariance selection
% Needs python 2.5 and cvxopt
% The covsel algorithm is described here
% http://abel.ee.ucla.edu/cvxopt/documentation/users-guide/node37.html
% http://abel.ee.ucla.edu/cvxopt/examples/documentation/chapter-6/

G = setdiag(G,1);
TT = tril(C .* G);
dlmwrite('input.data',TT,'delimiter', ' ');
!covselFile.py input.data output.data
precMat = dlmread('output.data');
precMat = mkSymmetric(precMat);
delete('input.data');
delete('output.data');
    
end
