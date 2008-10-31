function hh=polyFitErrorBars(varargin)
    [prior] = process_options(varargin, 'prior', 'mvnIG');
    [xtrain, ytrain, xtest, ytestNoisefree, ytestNoisy, sigma2] = polyDataMake(...
        'sampling', 'thibaux');
    degs = 1:2;
    for i=1:length(degs)
        deg = degs(i);
        T =  chainTransformer({rescaleTransformer, polyBasisTransformer(deg)});
        m = linregDist('transformer', T);
        m = inferParams(m, 'X', xtrain, 'y', ytrain, 'prior', prior, 'sigma2', sigma2);
        ypredTest = postPredict(m, xtest);
        figure;
        hold on;
        h = plot(xtest, mean(ypredTest),  'k-', 'linewidth', 3);
        %scatter(xtrain,ytrain,'r','filled');
        h = plot(xtrain,ytrain,'ro','markersize',14,'linewidth',3);
        NN = length(xtest);
        ndx = 1:20:NN;
        sigma = sqrt(var(ypredTest));
        mu = mean(ypredTest);
        [lo,hi] = credibleInterval(ypredTest);
        if strcmp(prior, 'mvn')
            % predictive distribution is a Gaussian
            assert(approxeq(2*1.96*sigma, hi-lo))
        end
        hh=errorbar(xtest(ndx), mu(ndx), mu(ndx)-lo(ndx), hi(ndx)-mu(ndx));
        %set(gca,'ylim',[-10 15]);
        set(gca,'xlim',[-1 21]);
    end
end