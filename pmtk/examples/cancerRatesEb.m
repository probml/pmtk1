%% Empirical Bayes Missouri Cancer Example
% Johnson and Albert  p67, p24
%#testPMTK
%% Setup Data
data.y = [0 0 2 0 1 1 0 2 1 3 0 1 1 1 54 0 0 1 3 0];
data.n = [1083 855 3461 657 1208 1025 527 1668 583 582 917 857 ...
    680 917 53637 874 395 581 588 383];
X = [data.y(:) data.n(:)-data.y(:)];    % EB matrix of counts
%% Create & Fit Distribution
dist = BetaBinomDist();
dist = fit(dist, 'data', X);
%% Extract Statistics
a = dist.a; b = dist.b;
display([a b]);
d = length(data.n); % ncities;
for i=1:d
    aPost(i) = a + data.y(i);
    bPost(i) = b + data.n(i) - data.y(i);
    thetaPostMean(i) = aPost(i)/(aPost(i) + bPost(i));
    thetaMLE(i) = data.y(i)/data.n(i);
end
thetaPooledMLE = sum(data.y)/sum(data.n);
%% Plot
figure;
subplot(4,1,1); bar(data.y); title('number of people with cancer (truncated at 5)')
set(gca,'ylim',[0 5])
subplot(4,1,2); bar(data.n); title('pop of city (truncated at 2000)');
set(gca,'ylim',[0 2000])
sf = 1000;
subplot(4,1,3); bar(thetaMLE*sf);title('MLE*1000 (red line=pooled MLE)');
hold on;h=line([0 20], sf*[thetaPooledMLE thetaPooledMLE]);
set(h,'color','r','linewidth',2)
subplot(4,1,4); bar(sf*thetaPostMean);title('posterior mean*1000 (red line=pop mean)')
popMean = a/(a+b);
hold on;h=line([0 20], sf*[popMean popMean]);
set(h,'color','r','linewidth',2)
if doPrintPmtk, doPrintPmtkFigures('cancerRatesEb'); end;


% 95% credible interval
figure; hold on;
for i=1:d
  q = betainv([0.025 0.975], aPost(i), bPost(i));
  h = line([q(1) q(2)], [i i]);
  median = betainv(0.5, aPost(i), bPost(i));
  h=plot(median,i,'*');
end
title('95% credible interval on theta, *=median')

fprintf('%4.4f ', thetaPostMean*1000); fprintf('\n')
if doPrintPmtk, doPrintPmtkFigures('cancerRatesCredibleEb'); end;