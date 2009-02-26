% tplot
%[X1,X2] = meshgrid(linspace(-5,5,100)',linspace(-5,5,100)');
[X1,X2] = meshgrid(linspace(-2,2,30)',linspace(-2,2,30)');
n = size(X1,1);
X = [X1(:) X2(:)];
%C = [1 .4; .4 1]; 
C = 0.1*eye(2);

figure(1);clf
df = 2;
p = mvtpdf(X,C,df);
surf(X1,X2,reshape(p,n,n));
title(sprintf('T distribution, dof %3.1f', df))

figure(2);clf
p = mvnpdf(X,[0 0],C);
surf(X1,X2,reshape(p,n,n));
title(sprintf('Gaussian'))
%set(gca,'zlim',[0 0.005])
