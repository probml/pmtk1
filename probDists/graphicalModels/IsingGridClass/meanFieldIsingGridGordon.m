function mu = meanFieldIsingGrid(J, CPDs, img, varargin)
% img should be an m*n  matrix 
% CPDs{1}=p(y|s=-1), CPDs{2}=p(y|s=1) 
% mu(i,j) = E(S(i,j)|y) where S(i,j) in {-1,1}

[maxIter, progressFn, parallelUpdates, inplaceUpdates, rate] = process_options(...
  varargin, 'maxiter', 100, 'progressFn', [], ...
  'parallelUpdates', false, 'inplaceUpdates', true, 'updateRate', 1);

[M,N] = size(img);
offState = 1; onState = 2;
logodds = logprob(CPDs{onState}, img(:)) - logprob(CPDs{offState}, img(:));
logodds = reshape(logodds, M, N);

% init
p1 = sigmoid(logodds);
mu = 2*p1-1;
mu = reshape(mu, M, N);
muNew  = mu;

% Shift operators - trick from
% http://www.cs.cmu.edu/~ggordon/variational/meanfield/meanfield.m
shu = spdiags(ones(N,1),1,N,N);
shd = spdiags(ones(N,1),-1,N,N);
shl = spdiags(ones(M,1),1,M,M);
shr = spdiags(ones(M,1),-1,M,M);

%% debug 
muP = tanh(0.5*logodds + J*(shl*mu + shr*mu + shu*mu + shd*mu));
muNew = zeros(size(mu));
for ix=1:N
  for iy=1:M
    pos = iy + M*(ix-1);
    neighborhood = pos + [-1,1,-M,M];
    neighborhood(([iy==1,iy==M,ix==1,ix==N])) = [];
    Sbar = J*sum(mu(neighborhood));
    muNew(pos) = tanh(Sbar + 0.5*logodds(pos));
  end
end
assert(approxeq(muP, muNew))

keyboard
 
for iter = 1:maxIter
  if parallelUpdates
    mu = (1-rate)*mu + rate*tanh(0.5*logodds + J*(shl*mu + shr*mu + shu*mu + shd*mu));
  else
    muNew = mu;
    for ix=1:N
      for iy=1:M
        pos = iy + M*(ix-1);
        neighborhood = pos + [-1,1,-M,M];
        neighborhood(([iy==1,iy==M,ix==1,ix==N])) = [];
        Sbar = J*sum(mu(neighborhood));
        if ~inplaceUpdates
          muNew(pos) = (1-rate)*muNew(pos) + rate*tanh(Sbar + 0.5*logodds(pos));
        else
          mu(pos) = (1-rate)*mu(pos) + rate*tanh(Sbar + 0.5*logodds(pos));
        end
      end
    end
    keyboard
    if ~inplaceUpdates, mu = muNew; end
  end
  if ~isempty(progressFn)
    feval(progressFn, mu, iter);
  end
end
 
end
