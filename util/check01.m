function y = check01(y)
% Ensure y(i) in {0,+1} 

u = unique(y);
S = setdiff(u, [0 1]);
if ~isempty(S)
  error('should only be 0,1')
end
