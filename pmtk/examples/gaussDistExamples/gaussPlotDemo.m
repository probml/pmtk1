%% Plot a Gaussian
xs = -3:0.01:3;
mu = 0; sigma2 = 1;
obj = gaussDist(mu, sigma2);
p = exp(logprob(obj,xs));
figure; plot(xs, p);
figure; plot(xs, normcdf(xs, mu, sqrt(sigma2)));