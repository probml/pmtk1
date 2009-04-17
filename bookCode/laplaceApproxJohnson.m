function [theta,C,logZ,iter] = laplaceApproxJohnson(logpost, start, varargin)
% f(i) = fn(params(i,:)) is unnormalized log posterior


[maxIter, thresh] = process_options(...
    varargin, 'maxIter', 10, 'thresh', 1e-3);

theta = start(:);
[f g H] = foo(start(:))
done = 0;
iter = 1;
while ~done & (iter < maxIter)
  [f g H] = foo(theta);
  %H = hessFn(theta);
  %g = gradFn(theta);
  C = -inv(H);
  thetaOld = theta;
  theta = theta + C*g(:); % Newton step
  if convergenceTest(norm(theta), norm(thetaOld), thresh)
    done = 1;
  else
    iter = iter + 1;
  end
end

ndim = length(theta); 
logZ = ndim/2*log(2*pi) + 0.5*logdet(C) + logpost(theta(:)');

  function [f,g,H] = foo(theta)
    % we will always call foo with a single column vector
    % but logpost expects row vectors (required by gradest)
    theta
    f = logpost(theta');
    g = gradest(logpost, theta')'; % grad est returns row vector, want column
    H = hessian(logpost, theta');
  end

end




