
% Test covariance selection on examples from Baback
randn('state', 0); 
d = 100;
sigma=randn(d); sigma=sigma'*sigma + diag(rand(d,1)); % random cov
nonZero = randn(d)>1; nonZero = (nonZero+nonZero')>0 ; % random G
S = sigma; G = setDiag(nonZero, 0);
assert(isposdef(S))
assert(isequal(G,G'))

methods = {};
methods{end+1} = 'covselPython';
methods{end+1} = 'covselMinfunc';


for m=1:length(methods)
  tic;
  precMat{m} = feval(methods{m}, S, G);
  t=toc;
  Ghat = precmatToAdjmat(precMat{m});
  valid = isequal(G, Ghat);
  figure;imagesc(precMat{m});
  ttl = sprintf('%s, pd %d, t %5.3f, valid %d', ...
    methods{m}, isposdef(precMat{m}), t, valid))
  title(ttl);
end



