function [mu,Sigma,logZ] = laplaceApprox(logpost, start)
% Compute a Gaussian approximation to the posterior
% logpost is the unnormalized log posterior
% logpost should take a matrix as input, each row is a different param
% vector, L(i) = log p(X(i,:))

options.Method = 'cg';
options.Display = 'off';
[f g H] = foo(start(:))
mu =  minFunc(@foo,start(:),options);
[f g H] = foo(mu(:));
ndim = length(mu); 
Sigma = -inv(H);
logZ = ndim/2*log(2*pi) + 0.5*logdet(Sigma) + logpost(mu(:)');

  function [f,g,H] = foo(theta)
    % minFunc will always call foo with a single column vector
    % but logpost expects row vectors
    theta
    f = logpost(theta');
    g = gradest(logpost, theta')'; % grad est returns row vector, want column
    H = hessian(logpost, theta');
  end

end


