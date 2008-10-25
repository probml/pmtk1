function [b,fb,funEvals] = mySearch(funObj,a,b,c,fa,fb,fc,varargin)
% a,b,c in ascending order, with fb < fa and fb < fc

optTol = 1e-4;
funEvals = 0;

while (c-a) > optTol && (max(fa,fc)-fb) > optTol
    %     fprintf('Iter = %d\n',funEvals);
    %     [a b c]
    %     pause;

    % Compute Mid-point of interval
    d = (c+a)/2;
    if abs(d-b) < optTol
        % Switch to Bisection if mid-point is too close to b
        if (c-b) > (b-a)
            d = (c+b)/2;
        else
            d = (a+b)/2;
        end
    end
 
    if d > b
        % Point in interval (b,c)
        fd = funObj(d,varargin{:});

        if fd > fb % Tigten interval to (a,d)
            red = c-d;
            c = d;
            fc = fd;
        else % Tighten interval to (b,c)
            red = b-a;
            a = b;
            fa = fb;
            b = d;
            fb = fd;
        end
    else
        % Point in interval (a,b)
        fd = funObj(d,varargin{:});

        if fd > fb % Tigten interval to (d,c)
            red = d-a;
            a = d;
            fa = fd;
        else % Tighten interval to (a,b)
            red = c-b;
            c = b;
            fc = fb;
            b = d;
            fb = fd;
        end
    end
    funEvals = funEvals+1;
end