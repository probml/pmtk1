function y = normalizeLogspace(x)
% Normalize in logspace while avoiding underflow.
% Each *row* of x is a log discrete distribution.
% y(i,:) = x(i,:) - logsumexp(x,2) = x(i) - log[sum_c exp(x(i,c)]
% eg post = exp(normalizeLogspace(logprior + loglik))

y = bsxfun(@minus, x, logsumexp(x,2));
 
end