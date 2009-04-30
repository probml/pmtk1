function linregRobustDemo()
%% linear regression where we minimize the L1 norm of the residuals using
%% linear programming
%#author John D'Errico
%#url http://www.mathworks.com/matlabcentral/fileexchange/loadFile.do?objectId=8553&objectType=FILE

seed = 0; setSeed(seed);
x = sort(rand(10,1));
y = 1+2*x + rand(size(x))-.5;
% add some outliers
x = [x' 0.1 0.5 1.0]';
y = [y' -3 -3 -3]';

figure;
plot(x,y,'ko','linewidth',2)
title 'Linear data with noise and outliers'

n = length(x);
% least squares soln
XX = [ones(n,1) x];
w = XX \ y;
hold on
xs = 0:0.01:1;
h(1)=plot(xs,w(1) + w(2)*xs,'r-','linewidth',2)

% L1 solution
f = [0 0 ones(1,2*n)]';
LB = [-inf -inf , zeros(1,2*n)];
UB = [];
Aeq = [ones(n,1), x, eye(n,n), -eye(n,n)];
beq = y;
params = linprog(f,[],[],Aeq,beq,LB,UB);
coef = params(1:2);

h(2) = plot(xs,coef(1) + coef(2)*xs,'b:','linewidth',2)
legend(h, 'L2', 'L1', 'location', 'northwest')
set(gca,'ylim',[-3.5 4])
end
