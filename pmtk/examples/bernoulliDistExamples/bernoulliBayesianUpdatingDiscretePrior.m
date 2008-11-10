%% Bayesian Updating with a Discrete Prior
Thetas = 0:0.1:1;
K = length(Thetas);
prior = DiscreteDist(normalize((1:K).^3), Thetas);
N1 = 3; N0 = 7; % data
%p = BinomDist(1, prior);
p = BernoulliDist(prior);
p = fit(p, 'suffStat', [N1 N1+N0],'method','bayesian');
post = p.mu;
ThetasDense = 0:0.01:1;
likDense = ThetasDense.^N1 .* (1-ThetasDense).^N0;
figure;
subplot(3,1,1); plot(prior); title('prior');
subplot(3,1,2); plot(ThetasDense, likDense); title('likelihood')
subplot(3,1,3); plot(post); title('posterior');