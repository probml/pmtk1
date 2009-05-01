setSeed(0);
d = 50;
%Sigma = randpd(d);
condnumber = 10; a = randn(d,1);
[Sigma] = covcond(condnumber,a);
cond(Sigma)
evalsTrue = sort(eig(Sigma),'descend');
mu = zeros(1,d);
f = [2 1/2 1/10];
for i=1:length(f)
  n = f(i)*d
  X = mvnrnd(mu,Sigma,n);
  Smle = cov(X);
  evalsMle = sort(eig(Smle),'descend');
  Sshrink = covshrinkKPM(X);
  evalsShrink = sort(eig(Sshrink),'descend');
  figure(i);clf; hold on
  ndx = 2:min(30,d);
  ndx = 1:d;
  if 0
    plot(evalsTrue(ndx), 'k-o', 'linewidth', 2);
    plot(evalsMle(ndx), 'b-x', 'linewidth', 2);
    plot(evalsShrink(ndx), 'r:s', 'linewidth', 2);
  else
    plot(log(evalsTrue(ndx)), 'k-o', 'linewidth', 2, 'markersize', 12);
    z=log(evalsMle(ndx));
    for ii=1:length(z), if ~isreal(z(ii)), z(ii)=nan; end; end
    plot(z, 'b-x', 'linewidth', 2, 'markersize', 12);
    plot(log(evalsShrink(ndx)), 'r:s', 'linewidth', 2, 'markersize', 12);
  end
  legend('true', 'mle', 'shrinkage')
  ylabel('log(eigenvalue)')
  title(sprintf('n=%d, d=%d, cond(MLE)=%g, cond(shrink)=%g', ...
		n, d, cond(Smle), cond(Sshrink)))
  if doPrintPmtk, printPmtkFigures(sprintf('covshrinkDemoLogN%d', n)); end;
end
  
