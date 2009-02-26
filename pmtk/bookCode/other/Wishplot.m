d = 2;
nus = [2 10];
doPrint = 0;
for j=1:2
  nu = nus(j);
  %S = 0.1*eye(2);
  S  = [4 3; 3 4];    % covariance
  seed = 0; rand('state', seed); randn('state', seed);
  figure(j);clf
  for i=1:9
    Lambda = wishrnd(S, nu);
    subplot(3,3,i)
    gaussPlot2d([0 0], Lambda);
    axis equal
  end
  suptitle(sprintf('Wishart(dof=%3.1f,S=[4 3; 3 4])', nu))
  fname = sprintf('C:/kmurphy/PML/Figures/wishplotS43nu%d.eps', nu)
  if doPrint, print(gcf, fname, '-depsc'); end
end
