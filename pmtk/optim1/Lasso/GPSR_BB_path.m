function [w, wdb]= GPSR_BB_path(X, y, taus, debias)

% calls GPSR_BB on a range of taus (strenght of L1 penalty)
% w(k,:) = solution with tau=taus(k)
% wdb(k,:) is the debiased solution; use this so that CV picks a better value
% Chooses values for tau if omitted

if nargin < 3 | isempty(taus)
  taumax = max(abs(X'*y)); % beyond this, w is all 0s
  taus = linspace(0.9,0,20) * taumax;
end
if nargin < 4, debias = false; end
npath = length(taus);
[taus, perm] = sort(taus,'descend'); % fastest to start with largest tau first
[n d] = size(X);
w = zeros(d, npath);
wdb = zeros(d, npath);
if debias
  [w(:,1), wdb(:,1)] = GPSR_BB(y, X, taus(1), 'initialization', 0, 'debias', 1, 'verbose', 0);
  for k=2:npath
    [w(:,k), wdb(:,k)] = GPSR_BB(y, X, taus(k), 'initialization', w(:,k-1), 'debias', 1, 'verbose', 0);
  end
else
  w(:,1) = GPSR_BB(y, X, taus(1), 'initialization', 0, 'debias',  0, 'verbose', 0);
  for k=2:npath
    [w(:,k)] = GPSR_BB(y, X, taus(k), 'initialization', w(:,k-1), 'debias', 0, 'verbose', 0);
  end
end
wdb = wdb';
w = w';
w = w(perm,:);
wdb = wdb(perm,:);