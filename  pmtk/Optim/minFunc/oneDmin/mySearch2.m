function [b,fb,funEvals] = mySearch(funObj,a,b,c,fa,fb,fc,varargin)
% a,b,c in ascending order, with fb < fa and fb < fc
% uses quadratic interpolation to generate trial points

optTol = 1e-4;
funEvals = 0;
while (c-a) > optTol && (max(fa,fc)-fb) > optTol
    %     fprintf('Iter = %d\n',funEvals);
    %     [a b c]
    %     pause;

    % Compute Quadratic Interpolate
     num = (fb-fc)*(b-a)^2 - (fb-fa)*(b-c)^2;
     den = (fb-fc)*(b-a) - (fb-fa)*(b-c);
     if den ~= 0
         d = b - (1/2)*num/den;
     else
         d = b;
     end

     if d > c || d < a
         d = b;
     end
     
     if abs(d-c) < optTol || abs(d-b) < optTol || abs(d-a) < optTol
         % Switch to mid-point if quadratic interpolate is too close any of
         % {a,b,c}
         d = (9/10)*d+(1/10)*(c+a)/2;
         if abs(d-b) < optTol
             d = (c+a)/2;
             if abs(d-b) < optTol
                 % Switch to Bisection if mid-point is too close to b
                 if (c-b) > (b-a)
                     d = (c+b)/2;
                 else
                     d = (a+b)/2;
                 end
             end
         end
     end
 
    if d > b
        % Point in interval (b,c)
        fd = funObj(d,varargin{:});

        if fd > fb % Tigten interval to (a,d)
            c = d;
            fc = fd;
        else % Tighten interval to (b,c)
            a = b;
            fa = fb;
            b = d;
            fb = fd;
        end
    else
        % Point in interval (a,b)
        fd = funObj(d,varargin{:});

        if fd > fb % Tigten interval to (d,c)
            a = d;
            fa = fd;
        else % Tighten interval to (a,b)
            c = b;
            fc = fb;
            b = d;
            fb = fd;
        end
    end
    funEvals = funEvals+1;
end