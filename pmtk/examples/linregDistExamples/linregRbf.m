%%  RBF Expansion
[xtrain, ytrain, xtest, ytest] = polyDataMake('sampling','thibaux');
lambda = 0.001; % just for numerical stability
sigmas = [0.05 0.5 50];
K = 10; % num centers
for i=1:length(sigmas)
    sigma = sigmas(i);
    T = ChainTransformer({RescaleTransformer, RbfBasisTransformer(K,sigma)});
    m  = LinregDist('transformer', T);
    m = fit(m, 'X', xtrain, 'y', ytrain, 'lambda', lambda);
    ypred = mean(predict(m, xtest));
    figure(1);clf
    scatter(xtrain,ytrain,'b','filled');
    hold on;
    plot(xtest, ypred, 'k', 'linewidth', 3);
    title(sprintf('RBF, sigma %f', sigma))

    % visualize the kernel centers
    [Ktrain,T] = train(T, xtrain);
    Ktest = test(T, xtest);
    figure(2);clf; hold on
    for j=1:K
        plot(xtest, Ktest(:,j));
    end
    title(sprintf('RBF, sigma %f', sigma))

    figure(3);clf;
    imagesc(Ktrain); colormap('gray')
    title(sprintf('RBF, sigma %f', sigma))
    %pause
end