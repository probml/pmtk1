function linregBasis
    demoBasisSimple();
    demoBasis();
    demoBasisDense();
    demoBasisSparse();
    
end



function demoBasisSimple()
        helperBasis(...
        'deg', 2, 'basis', 'quad', 'sampling', 'sparse', ...
        'plotErrorBars', true, 'prior', 'mvn');
end

function demoBasis()
    degs = [2 3];
    basis = {'quad', 'rbf'};
    sampling = {'sparse', 'dense'};
    for i=1:length(degs)
        for j=1:length(basis)
            for k=1:length(sampling)
                helperBasis(...
                    'deg', degs(i), 'basis', basis{j}, 'sampling', sampling{k}, ...
                    'plotErrorBars', true);
            end
        end
    end
end

function demoBasisDense()
    degs = [2 3];
    basis = {'quad', 'rbf'};
    sampling = {'dense'};
    for i=1:length(degs)
        for j=1:length(basis)
            for k=1:length(sampling)
                helperBasis(...
                    'deg', degs(i), 'basis', basis{j}, 'sampling', sampling{k}, ...
                    'plotErrorBars', false, 'plotSamples', false);
            end
        end
    end
end

function demoBasisSparse()
    degs = [2];
    basis = {'quad', 'rbf'};
    %basis = {'rbf'};
    sampling = {'sparse'};
    for i=1:length(degs)
        for j=1:length(basis)
            for k=1:length(sampling)
                helperBasis(...
                    'deg', degs(i), 'basis', basis{j}, 'sampling', sampling{k}, ...
                    'plotErrorBars', true, 'plotSamples', true);
            end
        end
    end
end

function helperBasis(varargin)
    [sampling, deg, basis, plotErrorBars, plotBasis, prior, plotSamples] = ...
        process_options(varargin, ...
        'sampling', 'sparse', 'deg', 3, 'basis', 'rbf', ...
        'plotErrorBars', true, 'plotBasis', true, ...
        'prior', 'mvn', 'plotSamples', true);

    [xtrain, ytrain, xtest, ytestNoisefree, ytestNoisy, sigma2] = polyDataMake(...
        'sampling', sampling, 'deg', deg);
    switch basis
        case 'quad'
            T = PolyBasisTransformer(2);
        case 'rbf'
            T = RbfBasisTransformer(10, 1);
    end
    m = LinregDist('transformer', T);
    m = fit(m,'X',xtrain,'y',ytrain,'prior',prior,'sigma2',sigma2);
    ypredTrain = predict(m,xtrain);
    ypredTest =  predict(m,xtest);
  

    figure; hold on;
    h = plot(xtest, mean(ypredTest),  'k-');
    set(h, 'linewidth', 3)
    h = plot(xtest, ytestNoisefree,'b-');
    set(h, 'linewidth', 3)
    %h = plot(xtrain, mean(ypredTrain), 'gx');
    %set(h, 'linewidth', 3, 'markersize', 12)
    h = plot(xtrain, ytrain, 'ro');
    set(h, 'linewidth', 3, 'markersize', 12)
    grid off
    legend('prediction', 'truth', 'training data', ...
        'location', 'northwest')
    if plotErrorBars
        NN = length(xtest);
        ndx = 1:3:NN;
        sigma = sqrt(var(ypredTest));
        mu = mean(ypredTest);
        h=errorbar(xtest(ndx), mu(ndx), 2*sigma(ndx));
        set(h, 'color', 'k');
    end
    title(sprintf('truth = degree %d, basis = %s, n=%d, prior %s', ...
        deg, basis, length(xtrain), prior));

    % superimpose basis fns
    if strcmp(basis, 'rbf') && plotBasis
        ax = axis;
        ymin = ax(3);
        h=0.1*(ax(4)-ax(3));
        [Ktrain,T] = train(T, xtrain);
        Ktest = test(T, xtest);
        K = size(Ktest,2); % num centers
        for j=1:K
            plot(xtest, ymin + h*Ktest(:,j));
        end
    end

    % Plot samples from the posterior
    if ~plotSamples, return; end
    figure;clf;hold on
    plot(xtest,ytestNoisefree,'b-','linewidth',3);
    h = plot(xtrain, ytrain, 'ro');
    set(h, 'linewidth', 3, 'markersize', 12)
    grid off
    nsamples = 10;
    for s=1:nsamples
        ws = sample(m.w);
        ms = LinregDist('w', ws(:), 'sigma2', sigma2, 'transformer', T);
        [xtrainT, ms.transformer] = train(ms.transformer, xtrain);
        ypred = mean(predict(ms, xtest));
        plot(xtest, ypred, 'k-', 'linewidth', 2);
    end
    title(sprintf('truth = degree %d, basis = %s, n=%d', deg, basis, length(xtrain)))
end
