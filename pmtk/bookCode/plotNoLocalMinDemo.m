% Ascher notes p4, fig 2.4 on p8
f = @(x,y) (x(:,1).^3 - x(:,1).^2 .* x(:,2) + 2 * x(:,2).^2);

[x,y] = meshgrid(-10:1:10);
z = reshape(f([x(:),y(:)]), size(x));
figure(1);clf
surf(x,y,z);
if doPrintPmtk, printPmtkFigures('ascherNoLocalMin'); end;
figure(2);clf
contour(x,y,z,[-100 -1:0.1:1 100]);

