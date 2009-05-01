figure(1);clf
%rhos=[0.99 0.9 0.8   0.7 0.5 0.3  0.2 0.1 0.01];
rhos=repmat([0.99 0.5 0.01]',1, 3); rhos=rhos(:)';
sy = 1;f =3;
sxx = [f*sy f*sy f*sy    sy sy sy    sy/f sy/f sy/f];
for i=1:length(rhos)
  rho = rhos(i);
  %sx = 1; sy = 3;
  %sx = 3; sy = 1;
  %sx = 1; sy = 1;
  sx = sxx(i);
  S = [sx^2 sx*sy*rho; sx*sy*rho sy^2];
  x = mvnrnd([0 0],S,5000);
  subplot(3,3,i)
  plot(x(:,1), x(:,2), '.');
  %title(sprintf('%s=%3.2f%s, %s=%3.2f', '\sigma_x', sx, '\sigma_y', '\rho', rho))
  title(sprintf('%s=%3.2f, %s=%3.2f', '\sigma_x', sx, '\rho', rho))
  axis([-8 8 -8 8])
  axis equal
  %axis off
end

  
if doPrintPmtk, doPrintPmtkFigures('corrcoef'); end;