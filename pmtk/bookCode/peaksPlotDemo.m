

%[X,Y,Z] = peaks; % 49x49 surface surface
[X,Y] = meshgrid(linspace(-2.5,2.5,40),linspace(-3,3,50));
Z = peaks(X,Y);

doSave = 0;

% find optimum by exhaustive search
Zmax = max(Z(:)); % 8.0752
[rowMax,colMax] = find(Zmax==Z); % max  row 38, col 25 
Zmin = min(Z(:)); % -6.54566
[rowMin,colMin] = find(Zmin==Z); % min  row 12, col 27
Xmax = X(rowMax,colMax); Ymax = Y(rowMax,colMax);
Xmin = X(rowMin,colMin); Ymin = Y(rowMin,colMin);
str = sprintf('max =%5.3f at x=%5.3f,y=%5.3f, min =%5.3f at x=%5.3f,y=%5.3f', ...
  Zmax, Xmax, Ymax, Zmin, Xmin, Ymin)

figure(1);clf; surf(X,Y,Z);
xlabel('x'); ylabel('y');
view(-19,48)
hold on; h=plot3(Xmax,Ymax,Zmax,'ro');
set(h,'markerfacecolor','r','markersize',12);
hold on; h=plot3(Xmin,Ymin,Zmin,'bo');
set(h,'markerfacecolor','b','markersize',12);
title(str)
if doSave, pdfcrop; print(gcf, '-dpdf', 'C:\kmurphy\PML\pdfFigures\peaksSurf.pdf'); end
if doPrintPmtk, printPmtkFigures('peaksSurf'); end;
  
figure(2);clf; contour(X,Y,Z);
xlabel('x'); ylabel('y'); colorbar;
if doSave, pdfcrop; print(gcf, '-dpdf', 'C:\kmurphy\PML\pdfFigures\peaksContour.pdf'); end
if doPrintPmtk, printPmtkFigures('peaksContour'); end;


figure(3);clf; imagesc(flipud(Z)); colorbar; 
xlabel('x'); ylabel('y');
if doSave, pdfcrop; print(gcf, '-dpdf', 'C:\kmurphy\PML\pdfFigures\peaksImagesc.pdf'); end
