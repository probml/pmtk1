%%  Demo of RBF Expansion for linear regression
[xtrain, ytrain, xtest, ytest] = polyDataMake('sampling','thibaux');
lambda = 0.001; % just for numerical stability
sigmas = [0.05 0.5 50];
K = 10; % num centers
for i=1:length(sigmas)
    sigma = sigmas(i);
    T = ChainTransformer({RescaleTransformer, RbfBasisTransformer(K,sigma)});
    %m  = LinregDist('transformer', T);
    %m = fit(m, 'X', xtrain, 'y', ytrain, 'prior', 'L2', 'lambda', lambda);
    m  = LinregL2('-transformer', T, '-lambda',lambda);
    m = fit(m, DataTable(xtrain,ytrain));
    ypred = mean(predict(m, xtest));
    figure;
    scatter(xtrain,ytrain,'b','filled');
    hold on;
    plot(xtest, ypred, 'k', 'linewidth', 3);
    title(sprintf('RBF, sigma %f', sigma))

    % visualize the kernel centers
    [Ktrain,T] = train(T, xtrain);
    Ktest = test(T, xtest);
    figure; hold on
    for j=1:K
        plot(xtest, Ktest(:,j));
    end
    title(sprintf('RBF, sigma %f', sigma))

    figure;
    imagesc(Ktrain); colormap('gray')
    title(sprintf('RBF, sigma %f', sigma))
    %pause
end