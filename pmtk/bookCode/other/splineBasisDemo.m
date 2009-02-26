% Based on code by John D'Erico
% http://www.mathworks.com/matlabcentral/fileexchange/loadFile.do?objectId=8553&objectType=fileY

if 0
  randn('state',0); rand('state', 0);
  n = 50;
  x = sort(rand(n,1));
  y = sin(pi*x) + 0.2*randn(size(x));
else
  makePolyData;
  x = rescaleData(xtrain);
  y = ytrain;
end

[X, knots] = splineBasis(x,100); % X(i,j) = 1 if x(i) is inside interval knot(j)
d = size(X,2);


figure(1);clf;plot(x,'o');hold on;
for i=1:d
  line([1 n], [knots(i),knots(i)])
end

figure(2);clf;spy(X)
