% Impute missing entries of the 4d Pima Indians data using an MVN  
%#author Cody Severinski

setSeed(0);
pima = csvread('./data/pima-r/pimatr.csv',1,0);
pima = pima(:,3:6);
pimaNan = pima;
[n,p] = size(pima);

probMissing = 0.3
miss = unifrnd(0,1,size(pima));
miss(miss < probMissing) = 0;
pimaNan(miss == 0) = NaN;
% pimaNanImg = pimaNan;
% pimaNanImg(miss == 0) = 0;

% Make a guess at what the model is
model = mkRndParams(MvnDist,p);
% But we know something of mu, Sigma from what we see
model.mu = nanmean(pimaNan)'; model.Sigma = nancov(pimaNan);

titles = {'glu','bp','skin','bmi'};

% So we can get the axis exactly right.
ax = {[50,200,0,40],[40,100,0,60],[0,100,0,60],[15,50,0,60]};

figure; subplot(4,4,1);
for row=1:p
for col=1:p
subplot(4,4,(row-1)*4+col);
if(row == col)
	hist(pimaNan(:,row)); title(titles{row}); axis(ax{row})
else
	scatter(pimaNan(:,row),pima(:,col))
end
end
end

% For comparison we also consider the 'best' model -- the model fitted given that we have access to all the data
fittedModel = {fit(model,'data',pima,'prior','none'), ...
	fit(model,'data',pima,'prior','niw'), ...
	fit(model,'data',pimaNan,'prior','none','fitArgs', {'verbose', true}), ...
	fit(model,'data',pimaNan,'prior','niw','fitArgs', {'verbose', true})};

% Impute the missing data and compute mse for each fitted model
for i = 1:length(fittedModel)
	pimaImputed{i} = impute(fittedModel{i},pimaNan);
	mse(i) = sum(sum(pimaImputed{i} - pima).^2)/prod(size(pimaImputed));
end

feature = {'glu', 'bp', 'skin', 'bmi'};

% Plot the imputed vs. the true values, with a 45 degree line to indicate "perfect" imputation
%for mod = 1:length(fittedModel)
figure();
for j = 1:p
subplot(2,2,j); hold on;
p1 = plot(pima(miss(:,j) == 0,j), pimaImputed{1}(miss(:,j) == 0,j),'b+');
xlabel(sprintf('true %s', feature{j})); ylabel(sprintf('imputed %s', feature{j}));
p2 = plot(pima(miss(:,j) == 0,j), pimaImputed{2}(miss(:,j) == 0,j),'r+');
xlabel(sprintf('true %s', feature{j})); ylabel(sprintf('imputed %s', feature{j}));
p3 = plot(pima(miss(:,j) == 0,j), pimaImputed{3}(miss(:,j) == 0,j),'bx');
xlabel(sprintf('true %s', feature{j})); ylabel(sprintf('imputed %s', feature{j}));
p4 = plot(pima(miss(:,j) == 0,j), pimaImputed{4}(miss(:,j) == 0,j),'rx');
xlabel(sprintf('true %s', feature{j})); ylabel(sprintf('imputed %s', feature{j}));
title(titles{j})
V = axis; lowLim = max(V(1),V(3)); upLim = min(V(2),V(4));
plot(lowLim:0.1:upLim, lowLim:0.1:upLim, 'g');
end
l = legend('All data, no prior', 'All data, niw prior', 'Visible data, no prior', 'Visible data, niw prior');
set(l,'Position',[0.45,0,0.1,0.05],'Orientation','horizontal','Box','off','FontSize',8);
%suplabel('Imputed vs real values')

% Plot differences
figure();
for j = 1:p
subplot(2,2,j); hold on;
plot(pima(miss(:,j) == 0,j) - pimaImputed{1}(miss(:,j) == 0,j),'b+');
plot(pima(miss(:,j) == 0,j) - pimaImputed{2}(miss(:,j) == 0,j),'r+');
plot(pima(miss(:,j) == 0,j) - pimaImputed{3}(miss(:,j) == 0,j),'bx');
plot(pima(miss(:,j) == 0,j) - pimaImputed{4}(miss(:,j) == 0,j),'rx');
end
subplot(2,2,1);
l = legend('All data, no prior', 'All data, niw prior', 'Visible data, no prior', 'Visible data, niw prior','location','SouthEastOutside')
set(l,'Position',[0.45,0,0.1,0.05],'Orientation','horizontal','Box','off','FontSize',8);
suplabel('Difference in imputed values (real - imputed)')

