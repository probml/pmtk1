function Hv = lbfgsHvFunc(v,old_dirs,old_stps,Hdiag)

S = old_dirs;
Y = old_stps;
k = size(Y,2);
L = zeros(k);
for j = 1:k
    L(j+1:k,j) = S(:,j+1:k)'*Y(:,j);
end
D = diag(diag(S'*Y));
N = [S/Hdiag Y];
M = [S'*S/Hdiag L;L' -D];

Hv = v/Hdiag - N*(M\(N'*v));