foo = load('yeastData310.mat'); % 'X' 310x7 'genes' 310x1 cell 'times 1x7
X = foo.X;
times = foo.times;
genes = foo.genes;
corrDist = pdist(X, 'corr');


figure;
clustergram(X,'RowLabels',genes, 'ColumnLabels',times)
title('hierarchical clustering')

figure;
dendrogram(linkage(corrDist, 'average'));
title('average link')
set(gca,'xticklabel','')

figure;
dendrogram(linkage(corrDist, 'complete'))
title('complete link')
set(gca,'xticklabel','')

figure;
dendrogram(linkage(corrDist, 'single'))
title('single link')

% Cut the tree to get 16 clusters
clusterTree = linkage(corrDist, 'average');
clusters = cluster(clusterTree, 'maxclust', 16);
figure;
for c = 1:16
    subplot(4,4,c);
    plot(times,X((clusters == c),:)');
    axis tight
end
suptitle('Hierarchical Clustering of Profiles')
