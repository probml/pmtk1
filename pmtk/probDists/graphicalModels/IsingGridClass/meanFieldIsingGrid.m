function X = meanFieldIsingGrid(J, CPDs, visVals, varargin)
% visVals should be an n*m matrix
% pon(i,j) = p(X(i,j)=1|y) where X(i,j) in {0,1}

[maxIter, progressFn, parallelUpdates] = process_options(...
  varargin, 'maxiter', 100, 'progressFn', [], 'parallelUpdates', false);

[M,N] = size(visVals);
Npixels = M*N;
offState = 1; onState = 2;
localEvidence = zeros(Npixels, 2);
for k=1:2
  localEvidence(:,k) = exp(logprob(CPDs{k}, visVals(:)));
end
logodds = log(localEvidence(:,onState)) - log(localEvidence(:,offState));

p1 = localEvidence( :, 2 ) ./ sum( localEvidence( :, 1:2 ), 2); % init
X = 2*p1-1;
X = reshape(X, M, N);
Xnew  = X;
for iter = 1:maxIter
  for ix=1:N
    for iy=1:M
      pos = iy + M*(ix-1);
      neighborhood = pos + [-1,1,-M,M];
      neighborhood(([iy==1,iy==M,ix==1,ix==N])) = [];
      Sbar = J*sum(X(neighborhood));
      if parallelUpdates
        Xnew(pos) = tanh(Sbar + 0.5*logodds(pos));
      else
        X(pos) = tanh(Sbar + 0.5*logodds(pos));
      end
    end
  end
  if parallelUpdates, X = Xnew; end
  if ~isempty(progressFn)
    feval(progressFn, X, iter);
  end
end
 
end
