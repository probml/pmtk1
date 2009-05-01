%% MCMC Sampling from a 2d Gaussians
% Compare MH and Gibbs
% We use a N(0, sigma*eye(2)) proposal and see the effect of changing sigma

Sigma = [1 -0.5; -0.5 1];
mu = [1; 1];
N = 500; 
model = MvnDist(mu, Sigma);
for i=1:2
  margExact{i} = marginal(model, i); %#ok
end


%targetFn = @(x) log(gausspdfUnnormalized(x, mu, Sigma));
targetFn = @(x) logprobUnnormalized(model,x);
xinit = mvnrnd(mu, Sigma);
S{1} = mhSample('target', targetFn, 'xinit', xinit, ...
  'Nsamples', N, 'proposal',  @(x) mvnrnd(x, 1*eye(2)));
S{2} = mhSample('target', targetFn, 'xinit', xinit, ...
  'Nsamples', N, 'proposal',  @(x) mvnrnd(x, 0.01*eye(2)));
fullCond = makeFullConditionals(model);
S{3} = gibbsSample(fullCond, xinit, N);
  
names= {'MH I', 'MH I .01 ', 'gibbs'};

for j=1:length(S)
    samples = S{j};
    ttl = names{j};
    figure;
    gaussPlot2d(mu, Sigma);
    hold on
    plot(samples(:,1), samples(:,2), '.');
    title(ttl)
    if doPrintPmtk, printPmtkFigures(sprintf('gauss2d%sSamples', strrep(strrep(names{j},' ',''),'.',''))); end;
    
    figure;
    samplesDist = SampleDist(samples, [1 2]); % convert raw samples to distribution
    for i=1:2
      margApprox{i} = marginal(samplesDist,i); %#ok
      subplot2(2,2,i,1);
      [h, histArea] = plot(margApprox{i}, 'useHisto', true);
      hold on
      [h, p] = plot(margExact{i}, 'scaleFactor', histArea, ...
        'plotArgs', {'linewidth', 2, 'color', 'r'});
      title(sprintf('exact m=%5.3f, v=%5.3f', mean(margExact{i}), var(margExact{i})));
      subplot2(2,2,i,2);
      plot(margApprox{i}, 'useHisto', false);
      title(sprintf('approx m=%5.3f, v=%5.3f', mean(margApprox{i}), var(margApprox{i})));
    end
    suptitle(ttl);
    if doPrintPmtk, printPmtkFigures(sprintf('gauss2d%sMarginals', strrep(strrep(names{j},' ',''),'.',''))); end;
    
    figure;
    for i=1:2
      subplot(1,2,i);
      stem(acf(samples(:,i), 30));
      title(ttl)
    end
end



