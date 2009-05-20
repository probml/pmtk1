function [f,g,H] = tracedObj(w,funObj,varargin)

global wValues
wValues(:,end+1) = w;

if nargout == 1
    f = funObj(w,varargin{:});
elseif nargout == 2
    [f,g] = funObj(w,varargin{:});
elseif nargout == 3
    [f,g,H] = funObj(w,varargin{:});
else
    [f,g,H,T] = funObj(w,varargin{:});
end