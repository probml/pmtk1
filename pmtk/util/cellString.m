function s = cellString(c)
% Given a cell array of strings, return a single string with all of the 
% cell entries concatinated horizontally, separted by commas. 
%
% Example
% s = cellString({'first','second','third','fourth','fifth','sixth'})
% s =
% first, second, third, fourth, fifth, sixth

   s = '';
   for i=1:numel(c)
      s = [s, c{i},', ']; %#ok
   end
    s(end-1:end) = [];
end