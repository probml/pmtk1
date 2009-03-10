function xs = gibbsMetrop(logpost, start, nsamples, scale, args)
% Gibbs sampling where we sample from each full conditional using Metropolis Hastings
% We assume the proposal is Gaussian(0, diag(scale))
% Returns xs(s, i) for s=1:nsamples, i=1:ndims
% Based on code by Johnson and Albert

p = length(start);
if nargin<3, nsamples = 1000; end
if nargin<4, scale=1*ones(1,p); end
if nargin<5, args={}; end
xs = zeros(nsamples, p);
x0 = start;
x1 = x0;
f0= feval(logpost, x0, args{:});
for s=1:nsamples
  for i=1:p
    x1(i) = x0(i) + randn*scale(i);
    f1 = feval(logpost, x1, args{:});
    u=rand < exp(f1-f0);
    if u==1
      x0 = x1;
      f0 = f1;
    end
    xs(s, i) = x0(i);
  end
end
  
