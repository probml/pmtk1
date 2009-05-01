% sigmoidPlot
figure(1); clf
xs = -20:0.1:20;
as = [0.3 1 3];
bs = [-10 0 10];
for i=1:3
  for j=1:3
    a = as(i); b = bs(j);
    ps = sigmoid(a*xs + a*b);
    subplot2(3,3,i,j)
    plot(xs, ps, 'k', 'linewidth', 2)
    hold on
    line([0 0], [0 1])
    title(sprintf('%s(%2.1fx + %2.1f)', '\sigma', a, a*b));
  end
end
if doPrintPmtk, doPrintPmtkFigures('sigmoidPlot'); end;

figure(2);clf
xs = -3:0.1:3;
ps = sigmoid(xs);
plot(xs,ps,'k-');
title('sigmoid')
figure(2);
ps = tanh(xs);
plot(xs,ps,'k-');
title('tanh')

