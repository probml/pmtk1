function X = genericGibbsSample(CPDs, x, Nburnin, Nsamples)
% X(i,:) = sample for i=1:n
% CPDs{j} is the full conditional for node j
d = length(x);
X = zeros(Nsamples, d);
keep = 1;
% find out which nodes are hidden
V = find(cellfun('isempty',CPDs));
H = setdiff(1:d, V);
for iter=1:(Nsamples + Nburnin)
  for i=H(:)' % only resample hidden nodes
    x(i) = feval(CPDs{i}, x);
  end
  if iter > Nburnin
    X(keep,:) = x; keep = keep + 1;
  end
end
end