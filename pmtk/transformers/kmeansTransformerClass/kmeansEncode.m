function assign = kmeansEncode(data, mu)
% data(i,:) i'th case
% mu(k,:)   k'th center (codebook vector)
% assign(i) in {1,...,K} is code index

dist = sqdist(data', mu');
[junk, assign] = min(dist,[],2);
