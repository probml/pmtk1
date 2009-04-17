function mhDemoJohnsonSmoking2()
% Johnson and Albert p58

doPlot = 0;
folder = 'C:/kmurphy/figures/other';

seed = 0; randn('state', seed); rand('state', seed);
%start = [2;5]';
start = [-5;10]';
numiter = 1000;
%xs = metrop(@logpostJohnsonRow, @proposal, start, numiter, {}, {[0.1 0.1]});type='bad';
xs = metrop(@logpostJohnsonRow, @proposal, start, numiter, {}, {sqrt([0.43 0.43])});type='good';

johnsonSmokingLogpostPlotExact();
hold on
plot(xs(:,1), xs(:,2), '.');
if doPlot, print(gcf,'-depsc',fullfile(folder,sprintf('mhDemoJohnson2_%s_samples.eps',type))), end

% trace plots
figure(2); plot(xs(:,1))
if doPlot, print(gcf,'-depsc',fullfile(folder,sprintf('mhDemoJohnson2_%s_trace.eps',type))), end
  
figure(3);
movavg = filter(repmat(1/50,50,1), 1, xs(:,1));
plot(movavg);
if doPlot, print(gcf,'-depsc',fullfile(folder,sprintf('mhDemoJohnson2_%s_smoothed_trace.eps',type))), end

% Auto correlation function
figure(4);clf
acf = acorr(xs(:,1),20);
stem(acf)
if doPlot, print(gcf,'-depsc',fullfile(folder,sprintf('mhDemoJohnson2_%s_acf.eps',type))), end

batchsizes = [1 2 4 8 16 32];
fprintf('%10s %10s %10s %10s\n', 'nbatches', 'batchsize', 'se', 'lag1corr');
for bi=1:length(batchsizes)
  batchsize = batchsizes(bi);
  [meanB, se, acf] = batchMeans(xs, batchsize);
  %fprintf('nbatches=%d, batchsize=%d, se=%5.3f, lag1 corr=%5.3f\n', ...
  %	  size(meanB,1), batchsize, se(1), acf(1))
  fprintf('%10d %10d %10.3f %10.3f\n', ...
	  size(meanB,1), batchsize, se(1), acf(1))
end
batchsize = 16;
[meanB, se, acf] = batchMeans(xs, batchsize);
meanAlpha = mean(meanB(:,1))
confIntLower = meanAlpha - 2*se(1);
confIntUpper = meanAlpha + 2*se(1);

%%%%%%%%

function xnew = proposal(xold, args)

sigmas = args;
xnew = [xold(1) + sigmas(1)*randn(1,1),...
	xold(2) + sigmas(2)*randn(1,1)];

function logp = logpostJohnsonRow(th)
logp = johnsonSmokingLogpost(th');
