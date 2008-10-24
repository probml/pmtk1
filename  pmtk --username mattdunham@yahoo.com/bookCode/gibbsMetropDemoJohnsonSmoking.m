function gibbsMetropDemoJohnson()
% Johnson and Albert p58

start = [2;5];
numiter = 5000;
%xs = gibbsMetrop(@logpostJohnson, start, numiter, [0.1 0.1]);
xs = gibbsMetrop(@logpostJohnson, start, numiter, sqrt([0.43 0.43]));

logpostJohnsonPlotExact();
hold on
plot(xs(:,1), xs(:,2), '.');




