function mhDemoGauss2d()
% Demo of Metropolis-Hastings algorithm for sampling from 
% a 2D Gaussian using a Gaussian proposal.
% Compare to gibbsGaussDemo.m

Nsamples = 2000;       
burnin  = 200; % 0; %1000;
folder = 'C:/kmurphy/figures/other';
doPrint= 0;

SigmaProps = {0.01*eye(2), [2 1; 1 1]};
%SigmaProps = { [2 1; 1 1]};
for trial=1:length(SigmaProps)
  SigmaProp = SigmaProps{trial};

mu = [0 0];
C = [2 1; 1 1];

targetArgs = {mu, C};
proposalArgs = {SigmaProp};

% try different starting seeds to check if mixing
seeds = [1 2 3];
samples = zeros(Nsamples-burnin, 2, length(seeds));
for c=1:length(seeds)
  seed = seeds(c);
  randn('state', seed); rand('state', seed);
  xinit = 20*rand(2,1); % initial state
  [tmp, naccept] = metrop(@target, @proposal, xinit, Nsamples,  targetArgs, proposalArgs);
  samples(:,:,c) = tmp(burnin+1:end,:);
end
  
% Trace plots
figure; colors = {'r', 'g', 'b', 'k'};
for c=1:length(seeds)
  plot(samples(:,1,c), colors{c});
  hold on
end
Rhat1 = EPSR(squeeze(samples(:,1,:)))
title(sprintf('sigmaProposal = %d, Rhat=%5.3f', trial, Rhat1))

if doPrint
  fname = fullfile(folder, sprintf('mhDemoGauss2d_trace%d.eps', trial))
  print(gcf, '-depsc', fname);
end

% Smoothed trace plots
figure; colors = {'r', 'g', 'b', 'k'};
for c=1:length(seeds)
  movavg = filter(repmat(1/50,50,1), 1, samples(:,1,c));
  plot(movavg,  colors{c});
  hold on
end
%title(sprintf('sigmaProposal = %3.2f', sigma));
title(sprintf('sigmaProposal = %d', trial));

if doPrint
  fname = fullfile(folder, sprintf('mhDemoGauss2d_smoothed_trace%d.eps', trial));
  print(gcf, '-depsc', fname);
end

% Plot auto correlation function
figure;
c=1;
acf = acorr(samples(:,1,c), 20);
stem(acf)
%title(sprintf('sigmaProposal = %3.2f', sigma))
title(sprintf('sigmaProposal = %d', trial));

if doPrint
  fname = fullfile(folder, sprintf('mhDemoGauss2d_acf%d.eps', trial));
  print(gcf, '-depsc', fname);
end

end % trial

return

figure;
%h=draw_ellipse(mu', C);
h = plot2dgauss(mu', C);
set(h, 'linewidth', 3, 'color', 'r');
axis equal
set(gca, 'xlim', [-5 5]);
set(gca, 'ylim', [-5 5]);
hold on
ndx = 1:10:size(samples,1); % only plot subset of points
plot(samples(ndx,1), samples(ndx,2), 'k.');

% Plot 1D exact and approximate marginals
for i=1:2
  figure;
  Nbins = 100;
  [h, xs] = hist(samples(:,1),Nbins);
  binWidth = xs(2)-xs(1);
  bar(xs, normalise(h)/binWidth);
  hold on
  ps = normpdf(xs, mu(i), sqrt(C(i,i)));
  plot(xs, ps, '-');
  title(sprintf('x%d', i))
end

%%%%%%%%%%

function  p = target(x, mu, Sigma)
p = log(mvnpdf(x(:)', mu, Sigma));

function xp = proposal(x, SigmaProp)
xp = mvnrnd(x, SigmaProp);


