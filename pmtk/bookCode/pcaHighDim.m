function [B, Xproj, mu] = pcaHighDim(X,K)
% Compute pca where d >> n
% Each row of X contains a feature vector, so X is n*d

[N d] = size(X);
mu = mean(X);
X = X - repmat(mu, N, 1);

if 1
  [U,S,V] = svd(X',0); % economy size
  %[U,S,V] = svds(X',K);  % slow!
  B = U(:,1:K);
else
  options.disp = 0; % stop display of intermediate resutls
  %[V,D] = eigs(X*X'/N, K, 'LM', options);  
  [V,D] = eig(X*X'/N);
  Eval = diag(D); 
  [junk,index]=sort(Eval); index=flipud(index); 
  V = V(:, index);
  D = diag(Eval(index));
  U = (X'*V)*inv(sqrt(N*D));
  B = U(:,1:K);
end
Xproj =  X*B;
