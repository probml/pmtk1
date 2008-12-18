%% Plot Demo
%#broken
for i=1:2
    if(i==1)
        useContour = false;
    else
        useContour = true;
    end
    figure;
    nu = [1 1 5 5];
    sigma2 = 2*[1 1 1 1];
    %sigma2 = [1 1 1 0.5];
    mu = [0 0 0 0];
    k =[1 10 1 10];
    %sigma2 = sigma2 .* nu;
    N = length(nu);
    [nr nc] = nsubplots(N);
    for i=1:N
        subplot(nr, nr, i);
        p = MvnInvWishartDist('mu',mu(i), 'Sigma',sigma2(i), ...
            'dof', nu(i), 'k', k(i));
        plot(p, 'xrange', [-1 1 0.1 2], 'useContour', useContour);
        shading interp;
        hold on
        str{i} = sprintf('NIW(%s=%3.1f, k=%3.1f, %s=%3.1f, %s=%3.1f)', ...
            'm', mu(i), k(i), '\nu', nu(i), 'S', sigma2(i));
        title(str{i})
        xlabel(sprintf('%s', '\mu'))
        ylabel(sprintf('%s', '\sigma^2'))
    end

end
