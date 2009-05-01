%% Sequential Bayesian Updating of a Bernoulli-Beta model. 
% In this example we draw samples from a BernoulliDist and then sequentially fit
% a Bernoulli_BetaDist model, plotting the posterior of the parameters at each
% iteration. 
%#testPMTK
%% Sample
setSeed(0);                        
m = BernoulliDist('-mu',0.7);        % 70% probability of success
n = 100;
X = sample(m, n);
%% Update & Plot
figure; hold on;
[styles, colors, symbols] =  plotColors();
ns = [0 5 50 100];
legendstr = cell(length(ns)+1,1);
for i=1:length(ns)
    n = ns(i);
    mm = Bernoulli_BetaDist('-prior',BetaDist(0.5,0.5));
    mm = fit(mm, 'data', X(1:n));
    plot(mm.muDist, 'plotArgs', {styles{i}, 'linewidth', 2});
    legendstr{i} = sprintf('n=%d', n);
end
box on;
xbar = mean(X);
pmax = 10;
h=line([xbar xbar], [0 pmax]); set(h, 'linewidth', 3,'Color','c');
legendstr{length(ns)+1} = 'truth';
legend(legendstr,'Location','NorthWest');
if doPrintPmtk, printPmtkFigures('betaSeqUpdate'); end;
