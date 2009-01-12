function [errRate,SE] = zeroOneErrorRate(yhat,ytest)
    
    err = reshape(yhat,size(ytest)) ~= ytest;
    errRate = mean(err);
    SE = std(err)/size(yhat,1);
end