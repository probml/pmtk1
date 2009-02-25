function precMat = covselChordalPython(C, G)
% Find MLE of precision matrix given graph G
% C = cov(data), G = adjacency matrix
% Same as covselPython, except exploits the graph structure
% Needs pychordal 

%#author Joachim Dahl
%#modified Kevin Murphy

G = setdiag(G,1);
TT = tril(C .* G);
dlmwrite('input.data',TT,'delimiter', ' ');
%!covselFastFile.py input.data output.data
system([' covselChordalPython.py ', ' input.data ',' output.data ',num2str(size(C,1))]);
precMat = dlmread('output.data');
precMat = mkSymmetric(precMat);
delete('input.data');
delete('output.data');
    

end