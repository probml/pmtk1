% studentTplot.m

figure(1);clf;
xs = -8:0.01:8;

mu0 = [0 0 0 0]; sigma20 = [1 1 1 3]; sigma0 = sqrt(sigma20);
nu0 = [1 3 10 1];

for i=1:4
  p = studentTpdf(xs(:), nu0(i), mu0(i), sigma0(i));
  subplot(2,2,i)
  plot(xs, p, 'linewidth', 2)
  str=sprintf('T(%s=%2.1f, %s=%2.1f, %s=%2.1f)',...
	      '\nu', nu0(i), '\mu_0', mu0(i),  '\sigma^2_0', sigma20(i));
  title(str)

  hold on
  p2 = normpdf(xs(:), mu0(i), sigma0(i));
  subplot(2,2,i)
  plot(xs, p2, 'r:', 'linewidth', 2)
end


nu = [0.1 1 5];
sigma = [1 1 1];
figure(2);clf
xs = -6:0.1:6;
styles = plotColors;
clear legendStr;
for i=1:length(nu)
  p = studentTpdf(xs(:), nu(i), 0, sigma(i));
  plot(xs, p, styles{i}); 
  hold on
  %legendStr{i} = sprintf('t(%s=%2.1f, %s=0,%s=%2.1f,%s=%2.1f)', ...
  %			 '\nu', nu(i), '\mu', '\sigma^2', sigma(i));
  legendStr{i} = sprintf('t(%s=%2.1f)', '\nu', nu(i));
end
title('Student T distributions')
hold on
plot(xs, normpdf(xs(:)', 0, sigma(1)), 'g:s');
legendStr{end+1} = 'N(0,1)';
legend(legendStr)
set(gca,'ylim',[-0.05 0.4]);
