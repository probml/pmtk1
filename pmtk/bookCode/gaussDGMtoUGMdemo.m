%% Convert Gaussian DGM to UGM to compare sparsity patterns



% chain
setSeed(0);
n = 5;
w=randn(n,1);
W = spdiags([w zeros(n,1) zeros(n,1)], -1:1, n, n);
T = eye(n)-W;
D = diag(ones(n,1));
K = T'*D*T;

% Diamond
setSeed(0);
n= 4;
W = zeros(4,4); % W(target, src)
w = rand(4,1);
W(2,1) = w(1); W(3,1) = w(2); W(4,2) = w(3); W(4,3) = w(4);
T = eye(n)-W;
D = diag(ones(n,1));
K = T'*D*T;
Sigma = inv(K);
