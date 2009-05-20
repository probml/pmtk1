function [xmin,fmin] = golden(funObj,a,b,c,fb)
% a,b,c in ascending order, with fb < fa and fb < fc

R = 0.61803399
C = 1-R;
optTol = 1e-4;

x0 = a;
x3 = c;

if abs(c-b) > abs(b-a)
    x1 = b;
    f1 = fb;
    x2 = b+C*(c-b);
    f2 = funObj(x2);
else
    x2 = b;
    f2 = fb;
    x1 = b-C*(b-a);
    f1 = funObj(x1);
end

i = 1;
while abs(x3-x0) > optTol*(abs(x1)+abs(x2))
    fprintf('Iteration %d\n',i);
    i = i + 1;
    
    if f2 < f1
        x0 = x1;
        x1 = x2;
        f1 = f2;
        x2 = R*x1+C*x3;
        f2 = funObj(x2);
    else
        x3 = x2;
        x2 = x1;
        f2 = f1;
        x1 = R*x2+C*x0;
        f1 = funObj(x1);
    end
end

if f1 < f2
    xmin = x1;
    fmin = f1;
else
    xmin = x2;
    fmin = f2;
end