
function X = meanFieldIsingGridBo(J, CPDs, visVals, varargin)
% visVars will be ignored
% visVals should be an n*m matrix

[maxIter, progressFn, thresh] = process_options(...
  varargin, 'maxiter', 100, 'progressFn', [], 'thresh', 1e-5);

[M,N] = size(visVals);
Npixels = M*N;
localEvidence = zeros(Npixels, 2);
for k=1:2
  localEvidence(:,k) = exp(logprob(CPDs{k}, visVals(:)));
end
offState = 1; onState = 2;
 
Q = localEvidence./repmat(sum(localEvidence,2), [1,2]);
Q = [Q; 0 0]; % add a pseudo node
PNODE = length(Q);
Unprocessed = true(M,N);
iter = 1;
while any(Unprocessed(1:end))
  ix = find(Unprocessed)';
  Nx = length(ix);
  [x y] = ind2sub([M N], ix);
  neighborhood = repmat(ix, 4, 1) + repmat([-1,1,-M,M]', 1,Nx);
  neighborhood(( [x==1; x==M; y==1; y==N] ) ) = PNODE;

  % compute update
  % un-normalized Q(x)
  unQx = [-1*ones(1, Nx);  ones(1, Nx)];
  Qdiff = Q(neighborhood,offState)- Q(neighborhood,onState);
  unQx = -J * unQx .* repmat( sum( reshape(Qdiff, [4 Nx]) ) , [2,1]);
  unQx = (exp(unQx)'.*localEvidence(ix,:));% transpose to Nx by again

  % normalize
  newQx = unQx./repmat(sum(unQx,2), 1,2);

  %       unmark processed nodes
  Unprocessed(ix) = false;
  % mark the neighbors of nodes with big changes unprocessed
  abs_change = abs(newQx(:,onState) - Q(ix, onState));
  Unprocessed(min(neighborhood(:,abs_change>thresh), N*M)) = true;

  Q(ix,:) = newQx;
  if ~isempty(progressFn)
    QQ = Q(1:end-1,onState); % remove pseudo node
    QQ = reshape(QQ, M, N);
    feval(progressFn, QQ, iter);
  end
  if iter > maxIter,break; else iter = iter + 1; end
end
Q = Q(1:end-1,:); % remove pseudo node
[junk, guess ] = max(Q,[],2);
X = zeros(M,N);
X((guess==offState)) = -1;
X((guess==onState)) = +1;

end
