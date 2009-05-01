function  demoTikhonov(lambda)

if nargin < 1, lambda = 10; end
% Tikhonov regularization for a piecewise smooth function
% Based on code by Uri Ascher
%N = 128;
N = 50;
noise = 0.5;
y = zeros(N,1);
h = 1/N;
for i=1:N
  x(i) = (i-1/2)*h;
  y(i) = fun(x(i));
end
% synthesize data by adding noise
randn('state',0);
y = y + randn(size(y))*noise;
% Construct first difference matrix
%D = sparse(N-1,N);
%for i=1:N-1
%  D(i,i) = -1;
%  D(i,i+1) = 1;
%end
D = spdiags(ones(N-1,1)*[-1 1], [0 1], N-1, N);
A = [speye(N); sqrt(lambda)*D];
b = [y; zeros(N-1,1)];
w = A \ b;
figure;
plot(x,y','bo');
hold on
plot(x,w','rx-','linewidth',2);
title(sprintf('noise=%3.1f, %s=%6.4f', noise, '\lambda', lambda));
if doPrintPmtk, doPrintPmtkFigures(sprintf('tikhonov%d', lambda)); end;
% nested fn
  function u = fun(z)
    if z < .25
      u = 1;
    elseif z < .5
      u = 2 ;
    elseif z < .7
      u = 2 - 100* (z - 0.5) * (0.7 - z);
    else
      u =  4;
    end
  end

end
    