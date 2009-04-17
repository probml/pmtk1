function M = removeRowsCols(M, rows, cols)
% Remove rows and columns from a matrix
% Example
% M = reshape(1:25,[5 5])
%> removeRowsCols(M, [2 3], 4)
%ans =
%     1     6    11    21
%     4     9    14    24
%     5    10    15    25
     
[nr nc] = size(M);

ndx = [];
for i=1:length(rows)
  tmp = repmat(rows(i), nc, 1);
  tmp2 = [tmp (1:nc)'];
  ndx = [ndx; tmp2];
end
for i=1:length(cols)
  tmp = repmat(cols(i), nr, 1);
  tmp2 = [(1:nr)' tmp];
  ndx = [ndx; tmp2];
end
if isempty(ndx), return; end
k = subv2ind([nr nc], ndx);
M(k) = [];
M = reshape(M, [nr-length(rows) nc-length(cols)]);
