function s = covType(Sigma)
if isscalar(Sigma)
  s = 'spherical';
elseif isvector(Sigma)
  s = 'diag';
else
  s = 'full';
end
end