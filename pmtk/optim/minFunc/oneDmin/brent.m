function [xmin,fmin] = brent(funObj,ax,bx,cx,fb)
% ax,bx,cx in ascending order, with fb < fa and fb < fc

optTol = 1e-4;
epsilon = 1e-10;
maxIter = 100;
gold = 0.381966;

e = 0;
a = ax;
b = cx;

x = bx;
fx = fb;
w = bx;
fw = fb;
v = bx;
fv = fb;

i = 1;
while i < maxIter
   i = i + 1;
   
   xm = .5*(a+b);
   tol1 = optTol*abs(x)+epsilon;
   tol2 = 2*tol1;
   if abs(x-xm) <= (tol2-.5*(b-a))
       xmin = x;
       fmin = fx;
       return;
   end
   if abs(e) > tol1
       r = (x-w)*(fx-fv);
       q = (x-v)*(fx-fw);
       p = (x-v)*q-(x-w)*r;
       q = 2*(q-r);
       if q > 0
           p = -p;
       end
       q = abs(q);
       etemp = e;
       e = d;
       if (abs(p) >= abs(.5*q*etemp)) || (p <= q*(a-x)) || (p >= q*(b-x))
           if x >= xm
               e = a-x;
           else
               e = b-x;
           end
           d = gold*e;
       else
           d = p/q;
           u = x+d;
           if (u-a < tol2) || (b-u < tol2)
               d = abs(tol1)*sign(xm-x);
           end
       end
   else
       if x >= xm
           e = a-x;
       else
           e = b-x;
       end
       d = gold*e;
   end
   if abs(d) >= tol1
       u = x+d;
   else
       u = x+sign(d)*abs(tol1);
   end
   fu = funObj(u);
   
   if fu <= fx
       if u >= x
           a = x;
       else
           b = x;
       end
       v = w;
       w = x;
       x = u;
       fv = fw;
       fw = fx;
       fx = fu;
   else
       if u < x
           a = u;
       else
           b = u;
       end
       if (fu <= fw) || (w == x)
           v = w;
           w = u;
           fv = fw;
           fw = fu;
       elseif (fu <= fv) || (v == x) || (v == w)
           v = u;
           fv = fu;
       end
   end
end
fprintf('Exceeded maxIter\n');
xmin = x;
fmin = fx;
           