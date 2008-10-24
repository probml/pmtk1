function mhDemoJohnsonSmoking()
% Johnson and Albert p58

start = [2;5]';
numiter = 1000;
%xs = metrop(@logpostJohnsonRow, @proposal, start, numiter, {}, {[0.1 0.1]});
xs = metrop(@logpostJohnsonRow, @proposal, start, numiter, {}, {sqrt([0.43 0.43])});

johnsonSmokingLogpostPlotExact();
hold on
plot(xs(:,1), xs(:,2), '.');

%%%%%%%%

function xnew = proposal(xold, args)

sigmas = args;
xnew = [xold(1) + sigmas(1)*randn(1,1),...
	xold(2) + sigmas(2)*randn(1,1)];

function logp = logpostJohnsonRow(th)
logp = johnsonSmokingLogpost(th');
