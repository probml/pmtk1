%% Plot Predictive Distribution of a KNN Classifer
%#testPMTK
load knnClassify3C
figure; hold on;
colors = {'or','ob','og'};
for c=1:3
   plot(Xtrain(ytrain==c,1),Xtrain(ytrain==c,2),colors{c},'LineWidth',2.5,'MarkerSize',8);
end
set(gca,'box','on','LineWidth',2);
title('Training Points','FontSize',12);
legend('Class 1','Class 2','Class 3','Location','NorthWest');

[X1,X2] = meshgrid(-5:0.1:7,-3:0.1:9);
[nrows,ncols] = size(X1);
K = 10;
beta = 0.3;
model = KnnDist('k',K,'beta',beta);
model = fit(model,Xtrain,ytrain);
p = mean(predict(model,[X1(:),X2(:)]))';
for c=1:3
    pc = reshape(p(:,c),nrows,ncols);
    figure;
    imagesc(X1(:),X2(:),pc);
    axis xy;
    set(gca,'box','on','LineWidth',2);
    title(sprintf('P( y = %d | X, D)\n(K = %d, beta = %2.1f)',c,K,beta),'FontSize',12)
    colorbar;
end

beta = 1;
model = fit(KnnDist('k',10,'beta',0.2,'localKernel','tricube'),Xtrain,ytrain);
p = mean(predict(model,[X1(:),X2(:)]))';
p1 = reshape(p(:,1),nrows,ncols);
figure;
imagesc(X1(:),X2(:),p1);
axis xy;
set(gca,'box','on','LineWidth',2);
title(sprintf('P( y = 1 | X, D)\n(K = %d, beta = %2.1f, tricube kernel)',K,beta),'FontSize',12)
colorbar;
placeFigures;