function se = stderr(x)
% Standard error of the mean
se = std(x) / sqrt(length(x));
end