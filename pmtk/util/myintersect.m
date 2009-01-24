function C = myintersect(A,B)
% MYINTERSECT Intersection of two sets of positive integers (much faster than built-in intersect)
% C = myintersect(A,B)

A = A(:)'; B = B(:)';

if isempty(A)
  ma = 0;
else
  ma = max(A);
end

if isempty(B)
  mb = 0;
else
  mb = max(B);
end

if ma==0 || mb==0
  C = [];
else
  %bits = sparse(1, max(ma,mb));
  bits = false(max(ma,mb),1);
  bits(A) = 1;
  C = B(bits(B));  
end

%sum( bitget( bitand( cliquesb(i), cliquesb(j) ), 1:52 ) ); 
