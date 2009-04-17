function str = csvstr(cellstring,delim)
% convert a cell array of strings to one long string with each entry 
% separated the specified delimiter, (default = ',').
    
    if ischar(cellstring)
       str = cellstring;
       return;
    end
    if isempty(cellstring)
       str = '';
       return;
    end
    
    if ~iscell(cellstring) || ~(allSameTypes({'',cellstring{:}}));
        error('csvstr input must be a cell array of strings');
    end
    if(nargin < 2), delim = ',';end
    str = cellstring{1};
    for i=2:numel(cellstring)
       str = [str,delim,cellstring{i}]; %#ok 
    end
    
    
end