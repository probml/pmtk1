function [a,b,c,fa,fb,fc,funEvals] = myBracket(funObj,options,b,fb,varargin)
% todo:
%   make initial scale depend on scaling of function
%   polynomial extrapolation

scale = myProcessOptions(options,'scale',1);

tPlus = scale;
tMinus = -scale;
plusDone = 0;
minusDone = 0;
funEvals = 0;
verbose = 0;

while 1
    if (~plusDone && rand < .5) || minusDone
        if verbose; fprintf('Testing Positive Move\n'); end
        c = tPlus;
        fc = funObj(c,varargin{:});
        funEvals = funEvals+1;
        if fc >= fb
            % c brackets a minimum in the plus direction
            plusDone = 1;
            if verbose; fprintf('Found Right Bracket\n'); 
            [b c]
            [fb fc]
            end
        elseif fc < fb
            % b brackets a minimum in the minus direction
            a = b;
            fa = fb;
            b = c;
            fb = fc;
            tPlus = tPlus*2;
            minusDone = 1;
            if verbose;
                fprintf('Found Left Bracket\n');
                [a b]
                [fa fb]
            end
        end
    else
        if verbose; fprintf('Testing Negative Move\n'); end
        a = tMinus;
        fa = funObj(a,varargin{:});
        funEvals = funEvals+1;
        
        if fa >= fb
            % a brackets a minimum in the minus direction
            minusDone = 1;
            if verbose; fprintf('Found Left Bracket\n');
            [a b]
            [fa fb]
             end
        elseif fa < fb
            % b brackets a minimum in the plus direction
            c = b;
            fc = fb;
            b = a;
            fb = fa;
            tMinus = tMinus*2;
            plusDone = 1;
            if verbose; fprintf('Found Right Bracket\n');
            [b c]
            [fb fc]
             end
        end
    end
    if verbose
    pause;
    end
    if plusDone && minusDone
        break;
    end
end
    %funEvals