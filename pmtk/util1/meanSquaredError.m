function err = meanSquaredError(y, yhat)

err = mean((y-yhat).^2);
