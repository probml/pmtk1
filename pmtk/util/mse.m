function [mu, se] = mse(y, yhat)
% mean squared error

err2 = (y-yhat).^2;
mu = mean(err2);
se = stderr(err2);
