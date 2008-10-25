function [visNodes, hidNodes] = findVisHid(x)  
visNodes = find(~isnan(x));
hidNodes = find(isnan(x));
end