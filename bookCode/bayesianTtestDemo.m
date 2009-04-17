function bayesfactorGeneDemo()

load('bayesFactorGeneData.mat');
ngenes = size(Xtreat,1);

% BF = p(data|H0)/p(data|H1) where H0=no change
for i=1:ngenes
  BF(i) = bayesianTtest(Xtreat(i,:), Xcontrol(i,:), 0, 100);
end
score = log(1./BF);

% pval = prob(>=data|H0), small pval means H0 unlikely
for i=1:ngenes
  [hyptest(i),pval(i)] = ttest(Xtreat(i,:), Xcontrol(i,:));
end
scoreFreq = 1./pval;

[faRateBF, hitRateBF, AUCBF] = ROCcurve(log(1./BF), truth, 0);
[faRatePval, hitRatePval, AUCPval] = ROCcurve(1./pval, truth, 0);

seed = 1; rand('state', seed);
R = rand(size(pval));
%R = 0.5*ones(size(pval));
[faRateRnd, hitRateRnd, AUCRnd] = ROCcurve(R, truth, 0);
[hitRateRnd2, faRateRnd2] = roc(truth,R);

figure(1);clf
h=plot(faRateBF, hitRateBF, 'b-'); set(h, 'linewidth', 3)
hold on
h=plot(faRatePval, hitRatePval, 'r:'); ; set(h, 'linewidth', 3)
h=plot(faRateRnd, hitRateRnd, 'k-.'); ; set(h, 'linewidth', 3)
h=plot(faRateRnd2, hitRateRnd2, 'c:'); ; set(h, 'linewidth', 3)
e = 0.05; axis([0-e 1+e 0-e 1+e])
xlabel('false alarm rate')
ylabel('hit rate')
grid on
legendstr{1} = sprintf('BF AUC=%5.3f', AUCBF);
legendstr{2} = sprintf('pval AUC=%5.3f', AUCPval);
legendstr{3} = sprintf('rnd AUC=%5.3f', AUCRnd);
legendstr{4} = sprintf('rnd2 AUC');
legend(legendstr)


figure(2);clf
perm = 1:ngenes;
%[junk, perm] = sort(truth);
doplot(Xcontrol, Xtreat, BF, pval, perm);

figure(3);clf
[junk, perm] = sort(truth);
doplot(Xcontrol, Xtreat, BF, pval, perm);



%%%%%%%%%%

function doplot(Xcontrol, Xtreat, BF, pval, perm)

%figure(1); clf;
subplot(2,2,1)
plot(Xcontrol(perm,:)); title('control')
subplot(2,2,2)
plot(Xtreat(perm,:)); title('treatment')
subplot(2,2,3)
scoreBF = log(1./BF);
plot(scoreBF(perm), '-'); title('log(BF(1,0))')
subplot(2,2,4)
scorePval = 1./pval;
plot(scorePval(perm)); title('1/pval');

%figure(2); clf
%imagesc([Xcontrol(perm,:) Xtreat(perm,:)])
