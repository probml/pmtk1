%joshCoins4

theta = 0.7; N = 5; alpha = 1;
alphaH = alpha; alphaT = alpha;
for i=1:(2^N)
  flips(i,:) = ind2subv(2*ones(1,N), i); % convert i to  bit vector
  Nh(i) = length(find(flips(i,:)==1));
  Nt(i) = length(find(flips(i,:)==2));
  nh = Nh(i); nt = Nt(i);
  margLik(i) = exp(betaln(alphaH+nh, alphaT+nt) - betaln(alphaH, alphaT));
  logBF(i) = betaln(alphaH+nh, alphaT+nt) - betaln(alphaH, alphaH) - N*log(0.5);
end

% sort in order of number of heads
[Nh, ndx] = sort(Nh);
margLik = margLik(ndx);
logBF = logBF(ndx);

figure(1); clf
hold on
p0 = (1/2)^N;
h=plot(margLik, 'o-');
h = line([0 2^N], [p0 p0]); set(h,'color','k','linewidth',3);
set(gca,'xtick', 1:2^N)
set(gca,'xticklabel',Nh)
xlabel('num heads')
title(sprintf('Marginal likelihood for Beta-Bernoulli model, %s p(D|%s) Be(%s|1,1) d%s', ...
	      '\int', '\theta', '\theta', '\theta'))
if doPrintPmtk, printPmtkFigures('joshCoins4'); end;

figure(2); clf
plot(exp(logBF), 'o-')
title('BF(1,0)')
set(gca,'xtick', 1:2^N)
set(gca,'xticklabel',Nh)
if doPrintPmtk, printPmtkFigures('joshCoins4BF'); end;
