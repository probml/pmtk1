function [a,b,c,fa,fb,fc] = bracket(funObj,a,b,fa,fb)
% Based on mnbrak function in numerical recipes

gold = 1.618034;
glimit = 100;
tiny = 1e-20;

if fb > fa
    [a,b,fa,fb] = deal(b,a,fb,fa);
end

c = b + gold*(b-a);
fc = funObj(c);

i = 1;
while fb > fc
    i=i+1;
    
    r = (b-a)*(fb-fc);
    q = (b-c)*(fb-fa);
    u = b-((b-c)*q-(b-a)*r)/(2*sign(q-r)*max(abs(q-r),tiny));
    ulim = b+glimit*(c-b);

    if (b-u)*(u-c) > 0
        fu = funObj(u);
        if fu < fc
            % Minimum between b and c
            a = b;
            fa = fb;
            b = u;
            fb = fu;
            return;
        elseif fu > fb
            % Minimum between a and u
            c = u;
            fc = fu;
            return;
        end
        % Interpolation did not work
        u = c+gold*(c-b);
        fu = funObj(u);
    elseif (c-u)*(u-ulim) > 0
        % Interpolation between c and limit
        fu = funObj(u);
        if fu < fc
            b = c;
            c = u;
            u = c+gold*(c-b);
            fb = fc;
            fc = fu;
            fu = funObj(u);
        end
    elseif (u-ulim)*(ulim-c) >= 0
        % Interpolation went past limit
        u = ulim;
        fu = funObj(u);
    else
        % Bad Interpolation
        u = c + gold*(c-b);
        fu = funObj(u);
    end
    a = b;
    b = c;
    c = u;
    fa = fb;
    fb = fc;
    fc = fu;
          

end

