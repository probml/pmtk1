function C = myintersect(A,B)
% MYINTERSECT Intersection of two sets of positive integers (much faster than built-in intersect)
% C = myintersect(A,B)

if isempty(A) || isempty(B)
    C = [];
else
    bits = false(max(max(A),max(B)),1);
    bits(A) = true;
    C = B(bits(B));
end
