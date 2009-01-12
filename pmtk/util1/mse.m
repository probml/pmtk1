function [mu, se] = mse(y, yhat)

err2 = (y-yhat).^2;
mu = mean(err2);
se = std(err2)/sqrt(length(err2));
