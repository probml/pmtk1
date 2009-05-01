function kmeansDemo

close all
doPrint = 1;

X = load('faithful.txt');
figure(1);clf; plot(X(:,1), X(:,2), '.', 'markersize', 10)
title('old faithful data')
grid on

if doPrint, pdfcrop; print_pdf('faithful'); end;
if doPrintPmtk, printPmtkFigures('faithful'); end;

seed = 4; rand('state', seed); randn('state', seed);

%figure(2);clf
K = 2;
[mu, assign, errHist] = kmeansSimple(X, K, 'fn', @plotKmeans, 'maxIter', 10);

figure(2)
if doPrint, pdfcrop; print_pdf('kmeansDemoFaithfulIter2'); end;
if doPrintPmtk, printPmtkFigures('kmeansDemoFaithfulIter2'); end;

%%%%%%

function plotKmeans(data, mu, assign, err, iter, converged)

fprintf('iteration %d, error %5.4f, converged %d\n', iter, err, converged);
mu
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
%folder = 'C:\kmurphy\PML\Figures';
%fname = sprintf('%s/kmeansDemoFaithfulIter%d.eps', folder, iter)
%if doPrint, print(gcf,'-depsc',fname); end
pause
end

end
