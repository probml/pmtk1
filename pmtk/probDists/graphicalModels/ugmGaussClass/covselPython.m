function precMat = covselPython(C, G)
% Find MLE precision matrix given covariance matrix C and GGM graph G
% aka covariance selection
% Needs python 2.5 and cvxopt
% The covsel algorithm is described here
% http://abel.ee.ucla.edu/cvxopt/documentation/users-guide/node37.html

%#author Joachim Dahl
%#url http://abel.ee.ucla.edu/cvxopt/examples/documentation/chapter-6/
%#modified Kevin Murphy

G = setdiag(G,1);
TT = tril(C .* G);
pp = fullfile(PMTKroot(), 'probDists\graphicalModels\ugmGaussClass');
curDir = pwd;
cd(pp);
dlmwrite('input.data',TT,'delimiter', ' ');
!covselPython.py input.data output.data
precMat = dlmread('output.data');
delete('input.data');
delete('output.data');
cd(curDir);
precMat = mkSymmetric(precMat);
    
end
