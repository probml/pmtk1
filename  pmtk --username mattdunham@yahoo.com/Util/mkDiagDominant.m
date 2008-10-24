function M = mkDiagDominant(M)

small = 0.1;
n = size(M,1);
for i=1:n
  s = sum(abs(M(i,[1:i-1 i+1:n])));
  M(i,i) = 1.01*s; % s + small;
end

assert(isposdef(M))
