%% Plot a Gaussian
xs = -3:0.01:3;
mu = 0; sigma2 = 1;
obj = GaussDist(mu, sigma2);
p = exp(logprob(obj,xs'));
figure; plot(xs, p,'LineWidth',2.5);
title('PDF');
if doPrintPmtk, doPrintPmtkFigures('gaussian1d'); end;
figure; plot(xs,cumsum(p),'LineWidth',2.5);
title('CDF');
if doPrintPmtk, doPrintPmtkFigures('gaussianCDF'); end;