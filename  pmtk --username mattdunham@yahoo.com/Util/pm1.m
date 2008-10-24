function y = pm1(y)
% Ensure y(i) in {-1,+1}  plus minus one
% Input can be in {0,1} or {1,2}

if ~isempty(find(y==0)) %0,1
  %y = 2*y-1;
  y(y==0) = -1;
else % 1,2
  y(y==2) = -1;
end
