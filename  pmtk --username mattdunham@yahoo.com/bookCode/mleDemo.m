mu = 3; s = 1;
setSeed(0);
D = randn(10,1) + mu;
xs = [0:0.01:7];
mus = [1 3  5];
figure; hold on
plot(D,0.01,'ro','markersize',12,'linewidth',3);
for i=1:length(mus)
  ps = normpdf(xs, mus(i), s);
  plot(xs, ps,'-');
end
xlabel('x')
ylabel('p(x|m)')

figure;
mus = linspace(-1,7,100);
for i=1:length(mus)
  l(i) = sum(log(normpdf(D,mus(i),s)));
end
plot(mus, l);
xlabel('mu')
ylabel('log p(D|mu)')

figure;
plot(mus, exp(l))
xlabel('mu')
ylabel('p(D|mu)')