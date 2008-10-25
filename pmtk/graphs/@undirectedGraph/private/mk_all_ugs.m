function Gs = mk_all_ugs(N, use_file)
% Generate all undirected graphs on N variables

% eg for N=4, there are 4.3/2 = 6 edges
% We reshape the bit vector as follows
% . . . .
% 1 . . .
% 2 3 . .
% 4 5 6 .

if nargin < 2, use_file = false; end

fname = sprintf('UGS%d.mat', N);
if use_file && exist(fname, 'file')
  S = load(fname, '-mat');
  fprintf('loading %s\n', fname);
  Gs = S.Gs;
  return;
end

nedges = nchoosek(N,2);
m = 2^nedges;
ind = ind2subv(2*ones(1,nedges), 1:m)-1; % all bit vectors
Gs = {};
for i=1:m
  G = zeros(N,N);
  bitv = ind(i,:);
  j = 1;
  for row=2:N
    for col=1:row-1
      G(row,col) = bitv(j);
      j = j + 1;
    end
  end
  G = mkGraphSymmetric(G);
  Gs{i} = G;
end

if use_file
  disp(['mk_all_ugs: saving to ' fname '!']);
  save(fname, 'Gs');
end
