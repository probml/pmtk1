function checkPM1(y)
% Check that y(i) = -1 or +1

if ~isempty(setdiff(unique(y), [-1 1]))
  error('use y=-1,+1')
end
