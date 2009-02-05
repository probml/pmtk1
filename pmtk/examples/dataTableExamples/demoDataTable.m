%% Simple Demonstration of the DataTable Class
%#testPMTK
function demoDataTable
n = 3; d = 2;
X = rand(n,d)
y = rand(n,1)
D = DataTable(X, y);
D(1:2)
D(1:2).X
D.X(1:2,:)
end