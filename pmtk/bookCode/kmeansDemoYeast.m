load('yeastData310.mat') % 'X', 'genes', 'times');

setSeed(0);
K = 16;
[cidx, ctrs] = kmeans(X, K, 'dist','corr','rep',5, 'disp','final');
% plot ecnetroids
figure;
for c = 1:16
    subplot(4,4,c);
    plot(times,X((cidx == c),:)');
    axis tight
end
suptitle('K-Means Clustering of Profiles');
if doPrintPmtk, printPmtkFigures('yeastKmeans16'); end;

figure;
for c = 1:16
    subplot(4,4,c);
    plot(times,ctrs(c,:)');
    axis tight
    axis off    % turn off the axis
end
suptitle('K-Means centroids')
if doPrintPmtk, printPmtkFigures('yeastKmeans16Centroids'); end;




