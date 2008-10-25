% rosenbrock argmin = (1,1) min = 0
% http://orion.uwaterloo.ca/~hwolkowi/henry/teaching/w06/666.w06/666miscfiles/extrosenfn.m
doSave = 1;
close all
%alpha=1;
alpha=100;
rosen = @(X) (1-X(:,1)).^2 + alpha*(X(:,2)-X(:,1).^2).^2;

%{
xx = [-2:0.125:2]';
yy = [-2:0.125:3]';
[x,y]=meshgrid(xx',yy') ;
meshd = alpha.*(y-x.*x).^2 + (1-x).^2; 
figure;
z=meshd;
%}

[xc,yc] = meshgrid(-1.5:.1:1.5);
zc = rosen([xc(:),yc(:)]);
zc = log(1+zc);
zc = reshape(zc,size(xc));
figure;
s=surf(xc,yc,zc);
view(-59,58);
%set(gca,'zlim',[0 500]);
hold on
%h=plot3(1,1,log(1+0),'ko');
%set(h,'markerfacecolor','k','markersize',12)
if doSave, pdfcrop; print(gcf, '-dpdf', 'C:\kmurphy\PML\pdfFigures\rosenSurf.pdf'); end

[xc,yc] = meshgrid(-2:.05:2);
zc = rosen([xc(:),yc(:)]);
zc = reshape(zc,size(xc));
figure;
contour(xc,yc,zc,[.1 1 4 16 64 256 1024 4096])
hold on
h=plot(1,1,'bo');
set(h,'markerfacecolor','b','markersize',12)
if doSave, pdfcrop; print(gcf, '-dpdf', 'C:\kmurphy\PML\pdfFigures\rosenContour.pdf'); end
