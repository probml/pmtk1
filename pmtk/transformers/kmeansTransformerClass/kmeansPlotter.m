function kmeansPlotter(data, mu, assign, err, iter)
[K D] = size(mu);
figure;
symbols = {'r.', 'gx', 'b', 'k'};
for k=1:K
  %subplot(2,2,iter)
  members = find(assign==k);
  plot(data(members,1), data(members, 2), symbols{k}, 'markersize', 10);
  hold on
  plot(mu(k,1), mu(k,2), sprintf('%sx', 'k'), 'markersize', 14, 'linewidth', 3)
  grid on
end
title(sprintf('iteration %d, error %5.4f', iter, err))
