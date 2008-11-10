%% Compare 4 Methods

methods = {'exact','gibbs','mhI','mhI01'};
for i=1:numel(methods)

    method = methods{i};
    % sample from a 2d Gaussian and compare to exact distribution
    m = MvnDist;
    setSeed(0);
    m.Sigma = [1 -0.5; -0.5 1];
    m.mu = [1; 1];
    %m = enterEvidence(m, [], []);
    for i=1:2
        margExact{i} = marginal(m,i);
    end
    N = 500;
    switch lower(method)
        case 'exact'
            m.stateInfEng = MvnExactInfer;
        case 'gibbs'
            m.stateInfEng = MvnMcmcInfer('method', 'gibbs', 'Nsamples', N);
        case 'mhi'
            m.stateInfEng = MvnMcmcInfer('method', 'mh', 'SigmaProposal', eye(2), ...
                'Nsamples', N, 'Nburnin', 500);
        case 'mhi01'
            m.stateInfEng = MvnMcmcInfer('method', 'mh', 'SigmaProposal', 0.01*eye(2), ...
                'Nsamples', N, 'Nburnin', 500);
        case 'mhtrue'
            m.stateInfEng = MvnMcmcInfer('method', 'mh', 'SigmaProposal', m.Sigma, ...
                'Nsamples', N, 'Nburnin', 500);
        otherwise
            error(['unknown method ' method])
    end
    %m = enterEvidence(m, [], []);
    X = sample(m, N);
    mS = SampleDist(X); % convert samples to a distribution
    for i=1:2
        margApprox{i} = marginal(mS,i);
    end
    figure;
    plot(m, 'useContour', 'true');
    hold on
    plot(mS);
    title(sprintf('%s', method))

    figure;
    plot(m, 'useContour', 'true');
    hold on
    plot(X(:,1), X(:,2), 'o-');
    title(method)

    figure;
    for i=1:2
        subplot2(2,2,i,1);
        [h, histArea] = plot(margApprox{i}, 'useHisto', true);
        hold on
        [h, p] = plot(margExact{i}, 'scaleFactor', histArea, ...
            'plotArgs', {'linewidth', 2, 'color', 'r'});
        title(sprintf('exact mean=%5.3f, var=%5.3f', mean(margExact{i}), var(margExact{i})));
        subplot2(2,2,i,2);
        plot(margApprox{i}, 'useHisto', false);
        title(sprintf('approx mean=%5.3f, var=%5.3f', mean(margApprox{i}), var(margApprox{i})));
    end

    % convergence diagnostics
    if ~strcmpi(method, 'exact')
        m.stateInfEng.seeds = 1:3;
        m.stateInfEng.Nburnin = 0;
        N = 1000;
        m.stateInfEng.Nsamples = N;
        % we must call enterEvidence since the params (Nsamples, seeds)
        % have changed; otherwise we will use the old values
        m = enterEvidence(m,[],[]);
        X  = sample(m, N); % X(sample, dim, chain)
        McmcInfer.plotConvDiagnostics(X, 1, sprintf('%s', method));
    end

end