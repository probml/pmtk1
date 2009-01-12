function tokens = tokenize(str,delimiter)
% tokenize a string    
    if(nargin < 2)
       delimiter = ' ' ; 
    end
    tokens = textscan(str,'%s','delimiter',delimiter);
    tokens = tokens{:};
end