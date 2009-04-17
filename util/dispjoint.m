function dispjoint(J)
% Display the entries in a joint probability table

sz = size(J);
for i=1:prod(sz)
  ndx = ind2subv(sz, i);
  fprintf(1, '%d ', ndx);
  fprintf(1, ': ');
  fprintf(1, '%6.4f ', J(i));
  fprintf(1, '\n');
end
