function [a,b,c,fa,fb,fc,funEvals] = myBracket(funObj,options,b,fb,varargin)
% Function for bracketing a 1D minimizer by quadratic interpolation

scale = myProcessOptions(options,'scale',1);
bracket = myProcessOptions(options,'bracket',2);

scale = max(scale,1e-4);
extrapTest = 0;

tPlus = scale;
tMinus = -scale;
plusDone = 0;
minusDone = 0;
funEvals = 0;
verbose = 0;

% First, move in either the positive or negative direction,
%   then move downhill from lower point
done = 0;
if rand < .5
    c = tPlus;
    fc = funObj(c,varargin{:});
    funEvals = funEvals + 1;
    if fc >= fb
        plusDone = 1;
        a = tMinus;
        fa = funObj(a,varargin{:});
        funEvals = funEvals + 1;
        if fa >= fb
            minusDone = 1;
        end
    else
        minusDone = 1;
        a = b;
        fa = fb;
        b = c;
        fb = fc;
        tPlus = tPlus*2;
        c = tPlus;
        fc = funObj(c,varargin{:});
        funEvals = funEvals+1;
        if fc >= fb
            plusDone = 1;
        end
    end
else
    a = tMinus;
    fa = funObj(a,varargin{:});
    funEvals = funEvals+1;
    if fa >= fb
        minusDone = 1;
        c = tPlus;
        fc = funObj(c,varargin{:});
        funEvals = funEvals+1;
        if fc >= fb
            plusDone = 1;
        end
    else
        plusDone = 1;
        c = b;
        fc = fb;
        b = a;
        fb = fa;
        tMinus = tMinus*2;
        a = tMinus;
        fa = funObj(a,varargin{:});
        funEvals = funEvals+1;
        if fa >= fb
            minusDone = 1;
        end
    end
end

while ~(plusDone && minusDone)

    % We have 3 points that do not bracket a minimum, use quadratic
    % interpolation

    xmin = tMinus*10;
    xmax = tPlus*10;
    d = polyinterp([a fa sqrt(-1);b fb sqrt(-1);c fc sqrt(-1)],0,xmin,xmax);

%     fd = funObj(d,varargin{:});
%     figure(2);clf;hold on
%     plot([a b c],[fa fb fc],'b.');
%     plot(d,fd,'r*');

    if d < a
        if minusDone
            fprintf('WTF!!!!\n');
            pause;
            tPlus = tPlus*2;
            d = tPlus;
            fd = funObj(d,varargin{:});
            funEvals = funEvals+1;
            if fd >= fc
                plusDone = 1;
            end
            a = b;
            fa = fb;
            b = c;
            fb = fc;
            c = d;
            fc = fd;
        else
            fprintf('Extrapolation\n');
            if extrapTest
                [a b c]
            d
            pause;
            end
            tMinus = max(tMinus*2,d);
            fd = funObj(d,varargin{:});
            funEvals = funEvals+1;
            if fd >= fc
                minusDone = 1;
            end
            c = b;
            fc = fb;
            b = a;
            fb = fa;
            a = d;
            fa = fd;
        end
    elseif d > c
        if plusDone
            tMinus = tMinus*2;
            fprintf('WTF!!!!\n');
            pause;
            d = tMinus;
            fd = funObj(d,varargin{:});
            funEvals = funEvals+1;
            if fd >= fa
                minusDone = 1;
            end
            c = b;
            fc = fb;
            b = a;
            fb = fa;
            a = d;
            fa = fd;
        else
            fprintf('Extrapolation\n');
            if extrapTest
                [a b c]
            d
            pause;
            end
            tPlus = min(tPlus*2,d);
            fd = funObj(d,varargin{:});
            funEvals = funEvals+1;
            if fd >= fc
                plusDone = 1;
            end
            a = b;
            fa = fb;
            b = c;
            fb = fc;
            c = d;
            fc = fd;
        end
    else % d between a and c
        fprintf('Interpolation\n');
        fd = funObj(d,varargin{:});
        funEvals = funEvals+1;
        if minusDone % a > b > c
            if d < b % d between a and b
                if fd > fb
                    fprintf('Interpolation not very helpful!\n');
                    a = d;
                    fa = fd;
                else
                    plusDone = 1;
                    c = b;
                    fc = fb;
                    b = d;
                    fb = fb;
                end
            else % d between b and c, with a > b > c
                if fd >= fb
                    plusDone = 1;
                    c = d;
                    fc = fd;
                else
                    if fd > fc
                        fprintf('Interpolation not very helpful!\n');
                    else
                        plusDone = 1;
                    end
                    a = b;
                    fa = fb;
                    b = d;
                    fb = fd;
                end
            end
        else % a < b < c
            if d < b % d between a and b
                if fd >= fb
                    minusDone = 1;
                    a = d;
                    fa = fd;
                else
                    if fd > fa
                        fprintf('Interpolation not very helpful!\n');
                    else
                        minusDone = 1;
                    end
                    c = b;
                    fc = fb;
                    b = d;
                    fb = fd;
                end
            else % d between b and c, with a < b < c
                if fd > fb
                    fprintf('Interpolation not very helpful!\n');
                    c = d;
                    fc = fd;
                else
                    minusDone = 1;
                    a = b;
                    fa = fb;
                    b = d;
                    fb = fd;
                end
            end
        end
    end
end
