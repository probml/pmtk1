% paretoPlot
figure(1);clf
k = [0.5 1 2 3];
legendStr = {};
styles = plotColors;
xs = sort([0.9:0.05:5]);
for i=1:length(k)
  p = paretopdf(xs, 1, k(i));
  h=plot(xs, p, styles{i});
  %h=plot(log(xs), log(p), styles{i});
  %set(h,'color',colors(i));
  hold on
  legendStr{i} = sprintf('%s=%2.1f', 'k', k(i));
end
legend(legendStr)
str=sprintf('Pa(m=1,k)');
title(str)
%xlabel('log x')
%ylabel('log p')
