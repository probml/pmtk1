function demoDataTable
%#demo

% We cannot include this as a method, since the D() syntax
% does not work unless you are outside the class
n = 3; d = 2;
X = rand(n,d)
y = rand(n,1)
D = dataTable(X, y);
D(1:2)
D(1:2).X
D.X(1:2,:)
end