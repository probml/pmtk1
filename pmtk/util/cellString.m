function s = cellString(c,delim)
% Converts a cell array of strings to a single string, (i.e. single-row
% character array). The specified delimiter, delim, is added between each
% entry. Include any spaces you want in delim. If delim is not specified,
% ', ' is used instead. If c is already a string, it is just returned. If c
% is empty, s = ''. 
%
% EXAMPLE:
%
% s = cellString({'touch /tmp/foo';'touch /tmp foo2';'mkdir /tmp/test'},' && ')
% s =
% touch /tmp/foo && touch /tmp foo2 && mkdir /tmp/test

   
   if ischar(c), s=c; return; end
   if isempty(c),s=''; return;end
   if nargin < 2, delim = ',  '; end
   s = '';
   for i=1:numel(c)
      s = [s, rowvec(c{i}),rowvec(delim)]; %#ok
   end
    s(end-numel(delim)+1:end) = [];
end