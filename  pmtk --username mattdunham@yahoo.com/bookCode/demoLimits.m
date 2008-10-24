function demoLimits

ns = 1:15;
figure(1);clf;hold on
for i=1:length(ns)
  ff(i) = f(ns(i));
  gg(i) = g(ns(i));
end

plot(ns, ff, 'r-x');
plot(ns, gg, 'g:*');

plot(ns, 2*gg, 'b--o');
plot(ns, 0.5*gg, 'k-.s');
grid on
legend('f', 'g', '2*g', '0.5 g')


function fn = f(n)
if iseven(n)
  fn = 2*n;
else
  fn =n;
end


function fn = g(n)
if iseven(n)
  fn = n;
else
  fn = 2*n;
end