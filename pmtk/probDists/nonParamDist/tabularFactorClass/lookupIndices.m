function ndx = lookupIndices(small, big)
% ndx(i) = location of small(i) in big
% e.g., small=[8,2], big=[2,4,8,7], ndx = [3 1]

if isempty(small)
  ndx = []; return;
end
ndx = zeros(length(small),1);
for i=1:length(small)
  ndx(i) = find(big==small(i));
end
