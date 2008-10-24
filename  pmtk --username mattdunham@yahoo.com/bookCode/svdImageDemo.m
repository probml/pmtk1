% based on Cleve Moler's book ch10 p25
clear all
doPrint = 0;
folder = 'C:\kmurphy\PML\Figures';
%load detail
load clown
%figure(1);clf
%image(X);colormap(gray(64));axis image; axis off
r = rank(X)
[U,S,V] = svd(X,0);
ranks = [1 2 5 10 20 r];
R = length(ranks);
%figure(2);clf
if 0
for i=1:R
  %subplot(2,2,i)
  %subplot(2,3,i)
  figure(i);clf
  k = ranks(i);
  Xhat = (U(:,1:k)*S(1:k,1:k)*V(:,1:k)');
  image(Xhat);colormap(gray(64));axis image; axis off
  title(sprintf('rank %d', k))
  %fname = sprintf('%s/svdImageDemoClown%d.eps', folder, k);
  fname = sprintf('%s/svdImageDemoClown%d.jpg', folder, k);
  if doPrint
    print(gcf, '-djpeg', fname)
  end
  pause
end
end

sigma = diag(S);
K = 100;
figure(1);clf
plot(log(sigma(1:K)), 'r-', 'linewidth', 3)
%ylabel(sprintf('%s_i','\sigma'))
ylabel(sprintf('log(%s_i)','\sigma'));
xlabel('i')
%fname = sprintf('%s/svdImageDemoClownSigma.eps', folder)



% scramble the data and replot (Hastie01 p491)
N = prod(size(X));
perm = randperm(N);
X2 = reshape(X(perm), size(X));
%X2 = randn(size(X));
%[n d] = size(X);
%X2 = X;
%for j=1:d
%  perm = randperm(n);
%  X2(:,j) = X(perm,j);
%end
[U,S2,V] = svd(X2,0);
sigma2 = diag(S2);
figure(1); hold on
plot(log(sigma2(1:K)), 'g:', 'linewidth', 3)
fname = sprintf('%s/svdImageDemoClownSigmaScrambled.eps', folder)
if doPrint
  print(gcf, '-depsc', fname)
end

legend('real data', 'randomized')
