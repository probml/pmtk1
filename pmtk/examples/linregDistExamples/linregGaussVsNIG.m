%% Compare Linreg_MvnDist to LinregMvnInvGammaDist
%#testPMTK
function linregGaussVsNIG()
    figure;
    h(1)= helperGaussVsNIG('prior', 'mvn', 'color', 'k');
    h(2)= helperGaussVsNIG('prior', 'mvnIG', 'color', 'r');
    legend(h, 'mvn', 'mvnIG')
end

function hh=helperGaussVsNIG(varargin)
    [prior, col] = process_options(varargin, 'prior', 'mvn', 'color', 'k');
    [xtrain, ytrain, xtest, ytestNoisefree, ytestNoisy, sigma2] = polyDataMake(...
        'sampling', 'sparse', 'deg', 2);
    T = PolyBasisTransformer(2);
    lambda = 1e-3;
    switch prior
      case 'mvn', m = Linreg_MvnDist('transformer', T, 'sigma2', sigma2,'priorStrength', lambda);
      case 'mvnIG', m = Linreg_MvnInvGammaDist('transformer', T, 'priorStrength', lambda);
    end
    m = fit(m, 'X', xtrain, 'y', ytrain);
   
    ypredTest = predict(m, xtest);

    hold on;
    h = plot(xtest, mean(ypredTest),  'k-');
    set(h, 'linewidth', 3)
    h = plot(xtest, ytestNoisefree,'b-');
    set(h, 'linewidth', 3)
    h = plot(xtrain, ytrain, 'ro');
    set(h, 'linewidth', 3, 'markersize', 12)
    grid off
    if 1
        NN = length(xtest);
        ndx = 1:3:NN;
        sigma = sqrt(var(ypredTest));
        mu = mean(ypredTest);
        hh=errorbar(xtest(ndx), mu(ndx), 2*sigma(ndx));
        set(hh, 'color', col);
    end
end