function M = mkDiagDominant(M)

d = size(M,1);
for i=1:d
  s = sum(M(i,:)) - M(i,i); % sum of row i except diagonal
  M(i,i) = 1.01*s; % make diag 1% larger than sum
end
assert(isposdef(M))