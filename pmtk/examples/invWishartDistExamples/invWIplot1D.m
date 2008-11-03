%% Plot Some invchi2 Distributions
figure;
%nu = [1 1 1  5 5 5 ];
nu = [1 1 1 2 2 2];
sigma2 = [0.5 1 2  0.5 1 2];
%sigma2 = sigma2 .* nu;
N = length(nu);
[styles, colors, symbols] =  plotColors;
%colors = hsvrand(N);
xrange =[0.01 2];
for i=1:N
    h = plot(invWishartDist(nu(i), sigma2(i)), 'plotArgs', styles{i}, 'xrange',xrange);
    hold on
    str{i} = sprintf('%s %3.1f, %s %3.1f', '\nu', nu(i), '\sigma^2', sigma2(i));
end
legend(str);
title(sprintf('IW(%s, %s)', '\nu', '\sigma^2'));