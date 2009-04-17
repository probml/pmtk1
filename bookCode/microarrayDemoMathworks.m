%http://www.mathworks.com/access/helpdesk/help/toolbox/bioinfo/index.html?/access/helpdesk/help/toolbox/bioinfo/ug/a1060813239b1.html


load yeastdata.mat

%%%%%%%%%%%%% filtering
emptySpots = strcmp('EMPTY',genes);
yeastvalues(emptySpots,:) = [];
genes(emptySpots) = [];

nanIndices = any(isnan(yeastvalues),2);
yeastvalues(nanIndices,:) = [];
genes(nanIndices) = [];

mask = genevarfilter(yeastvalues);
% Use the mask as an index into the values to remove the 
% filtered genes.
yeastvalues = yeastvalues(mask,:);
genes = genes(mask);

[mask, yeastvalues, genes] = genelowvalfilter(yeastvalues,genes,'absval',log2(4));
[mask, yeastvalues, genes] = geneentropyfilter(yeastvalues,genes,'prctile',15);

numel(genes) % 310

X = yeastvalues;
save('C:\kmurphy\PML\Data\Data\yeastData310.mat', 'X', 'genes', 'times');


%%%%%%%%%%% Clustering

corrDist = pdist(yeastvalues, 'corr');
clusterTree = linkage(corrDist, 'average');
clusters = cluster(clusterTree, 'maxclust', 16);
figure
for c = 1:16
    subplot(4,4,c);
    plot(times,yeastvalues((clusters == c),:)');
    axis tight
end
suptitle('Hierarchical Clustering of Profiles');


[cidx, ctrs] = kmeans(yeastvalues, 16,... 
                      'dist','corr',...
                      'rep',5,...
                      'disp','final');
figure
for c = 1:16
    subplot(4,4,c);
    plot(times,yeastvalues((cidx == c),:)');
    axis tight
end
suptitle('K-Means Clustering of Profiles');


figure
for c = 1:16
    subplot(4,4,c);
    plot(times,ctrs(c,:)');
    axis tight
    axis off    % turn off the axis
end
suptitle('K-Means Clustering of Profiles');


figure
clustergram(yeastvalues(:,2:end),'RowLabels',genes, 'ColumnLabels',times(2:end))

X= yeastvalues;
figure(4);clf;imagesc(X);colormap(redgreencmap)


%%%%%%%%%%%%% PCA

[pc, zscores, pcvars] = princomp(yeastvalues)

cumsum(pcvars./sum(pcvars) * 100)


figure; plot(X')
figure; plot(pc)


figure
scatter(zscores(:,1),zscores(:,2));
xlabel('First Principal Component');
ylabel('Second Principal Component');
title('Principal Component Scatter Plot');


figure
pcclusters = clusterdata(zscores(:,1:2),6);
gscatter(zscores(:,1),zscores(:,2),pcclusters)
xlabel('First Principal Component');
ylabel('Second Principal Component');
title('Principal Component Scatter Plot with Colored Clusters');
