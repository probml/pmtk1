function p = studentTpdf(xs, nu, mu, sigma2)
% mu = mean, sigma = std, nu = dof


%sigma2 = sigma^2;
sigma = sqrt(sigma2);
p = (1/sigma)*tpdf( (xs-mu) ./ sigma, nu);

tmp = ((xs-mu).^2) ./  (nu*sigma2);
p2 = gamma(nu/2+0.5)/gamma(nu/2) * (sigma2*pi*nu)^(-0.5) * (1+tmp).^(-(nu+1)/2);
assertKPM(approxeq(p,p2))
