function d = myndims(M)
% MYNDIMS Like the built-in ndims, except handles [] and vectors properly
% 
% The behavior is best explained by examples
% - M = [],          myndims(M) = 0,    ndims(M) = 2
% - M = rand(1,1),   myndims(M) = 1,    ndims(M) = 2
% - M = rand(2,1),   myndims(M) = 1,    ndims(M) = 2
% - M = rand(1,2,1), myndims(M) = 2,    ndims(M) = 2
% - M = rand(1,2,2), myndims(M) = 3,    ndims(M) = 3

if isempty(M)
  d = 0;
elseif isvector(M)
  d = 1;
else
  d = ndims(M);
end
