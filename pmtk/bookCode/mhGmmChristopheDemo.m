function demoMhGmmChristoph()

% Target = mixture of two 1d Gaussians
% MH Proposal = Gaussian kernel
% Show 3d plot of samples and their histogram
% Written by Christoph Andrieu
% Modified by Kevin Murphy

doPrint = 0;
sigmas = [10 100 500];
sigmas = [10];
for i=1:length(sigmas)
  helper(sigmas(i), doPrint)
end

function helper(sigma_prop, doPrint)

seed = 0; setSeed(seed);

nb_iter = 1000;
nb_iter  = 50

lambda = .3;
mean_1 = -50;
mean_2 = 50;
sigma_mc_1 = 10;
sigma_mc_2 = 10;

x = zeros(nb_iter, 1);
x(1) = mean_2;
u = rand(nb_iter, 1);

for iter = 2:nb_iter
  z = sigma_prop * randn(1, 1);
  x_prop = x(iter-1) + z;
  alpha = lambda * exp(-.5 * (x_prop - mean_1).^2 / sigma_mc_1^2) + ...
    (1 -lambda)* exp(-.5 * (x_prop - mean_2).^2 / sigma_mc_2^2);
  alpha = alpha / (lambda * exp(-.5 * (x(iter-1) - mean_1).^2 / sigma_mc_1^2) +...
                     (1 -lambda)* exp(-.5 * (x(iter-1) - mean_2).^2 / sigma_mc_2^2));
  if(u(iter)<alpha)
    x(iter) = x(iter -1) + z;
  else
    x(iter) = x(iter-1);
  end
end

x(:)'
return

figure;
% evaluate target density on a dense grid
x_real = linspace(-100, 100, nb_iter);
y_real = (lambda * exp(-.5 * (x_real - mean_1).^2 / sigma_mc_1^2)/sigma_mc_1 + ...
  (1 -lambda)* exp(-.5 * (x_real - mean_2).^2 / sigma_mc_2^2)/sigma_mc_2)/sqrt(2*pi);


Nbins = 100;
plot3(1:nb_iter, x, zeros(nb_iter, 1))
hold on
plot3(ones(nb_iter, 1), x_real, y_real)
[u,v] = hist(x, linspace(-100, 100, Nbins));
plot3(zeros(Nbins, 1), v, u/nb_iter*Nbins/200, 'r')
hold off
grid
view(60, 60)

xlabel('Iterations')
ylabel('Samples')
title(sprintf('MH with N(0,%5.3f^2) proposal', sigma_prop))

if doPrint
  folder = 'C:\kmurphy\PML\pdfFigures';
  pdfcrop;
  fname = fullfile(folder, sprintf('demoMhGmmChristoph%d', sigma_prop))
  print(gcf, '-dpdf', fname);
end