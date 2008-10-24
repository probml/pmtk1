function decisionBoundariesDemoSurf
% Decision boundaries induced by a mixture of two 2D Gaussians
% Based on code by Tommi Jaakkola

folder = 'C:/kmurphy/figures/other';
save = 0;

p1 = 0.5; p2 = 1-p1;
mu1 = [1 1]'; mu2 = [-1 -1]';
S1 = eye(2); S2 = eye(2);
plotgaussians(p1, mu1, S1, p2, mu2, S2, 'linear boundary');
if save, figure(1); print(gcf, fullfile(folder, 'dboundariesContour1.eps'), '-depsc'); end
if save, figure(2); print(gcf, fullfile(folder, 'dboundariesSurf1.eps'), '-depsc'); end
pause

p1 = 0.995; p2 = 1-p1;
mu1 = [1 1]'; mu2 = [-1 -1]';
S1 = eye(2); S2 = eye(2);
plotgaussians(p1, mu1, S1, p2, mu2, S2, 'linear boundary, both means on same side');
if save, figure(1); print(gcf, fullfile(folder, 'dboundariesContour2.eps'), '-depsc'); end
if save, figure(2); print(gcf, fullfile(folder, 'dboundariesSurf2.eps'), '-depsc'); end
pause

p1 = 0.5; p2 = 1-p1;
mu1 = [1 1]'; mu2 = [-1 -1]';
S1 = [2 0; 0 1]; S2 = eye(2);
plotgaussians(p1, mu1, S1, p2, mu2, S2, 'parabolic boundary');
if save, figure(1); print(gcf, fullfile(folder, 'dboundariesContour3.eps'), '-depsc'); end
if save, figure(2); print(gcf, fullfile(folder, 'dboundariesSurf3.eps'), '-depsc'); end
pause

p1 = 0.5; p2 = 1-p1;
mu1 = [0 0]'; mu2 = [0 0]';
S1 = [4 0; 0 2]; S2 = [1 0; 0 2];
plotgaussians(p1, mu1, S1, p2, mu2, S2, 'disconnected regions');
if save, figure(1); print(gcf, fullfile(folder, 'dboundariesContour4.eps'), '-depsc'); end
if save, figure(2); print(gcf, fullfile(folder, 'dboundariesSurf4.eps'), '-depsc'); end
pause

p1 = 0.5; p2 = 1-p1;
mu1 = [0 0]'; mu2 = [0 0]';
S1 = [1 0; 0 1]; S2 = [2 0; 0 2];
plotgaussians(p1, mu1, S1, p2, mu2, S2, 'circular boundary');
if save, figure(1); print(gcf, fullfile(folder, 'dboundariesContour5.eps'), '-depsc'); end
if save, figure(2); print(gcf, fullfile(folder, 'dboundariesSurf5.eps'), '-depsc'); end
pause

p1 = 0.5; p2 = 1-p1;
mu1 = [-2 1]'; mu2 = [1 1]';
S1 = [1 0; 0 0.5]; S2 = [2 0; 0 4];
plotgaussians(p1, mu1, S1, p2, mu2, S2, 'skewed elliptical boundary, only one mean inside');
if save, figure(1); print(gcf, fullfile(folder, 'dboundariesContour6.eps'), '-depsc'); end
if save, figure(2); print(gcf, fullfile(folder, 'dboundariesSurf6.eps'), '-depsc'); end
pause


%%%%%%%%%%
  
function h = plotgaussians(p1,mu1,S1,p2,mu2,S2, str)

figure(1);clf
%[x,y] = meshgrid(linspace(-10,10,100), linspace(-10,10,100));
[x,y] = meshgrid(linspace(-10,10,50), linspace(-10,10,50));
[m,n]=size(x);
X = [reshape(x, n*m, 1) reshape(y, n*m, 1)];
g1 = reshape(mvnpdf(X, mu1(:)', S1), [m n]);
g2 = reshape(mvnpdf(X, mu2(:)', S2), [m n]);
hold;
contour(x,y,g1, 'r:');
contour(x,y,g2, 'b--');
[cc,hh]=contour(x,y,p1*g1-p2*g2,[0 0], '-k');
set(hh,'linewidth',3);
axis equal
title(str)

figure(2);clf
%surfc(x, y, p1*g1);
surfc(x, y, g1);
hold
surfc(x, y, g2);
view(-10,50)
title(str)
