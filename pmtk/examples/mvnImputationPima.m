% Impute missing entries of the 4d Pima Indians data using an MVN  
%#author Cody Severinski

setSeed(0);
pima = csvread('pimatr.csv',1,0);
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
model = mkRndParams(MvnDist(),p);
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
fittedModel{1} = fit(model,'data',pima,'prior','none');
fittedModel{2} = fit(model,'data',pimaNan,'prior','none','fitArgs', {'verbose', true});


% Impute the missing data and compute mse for each fitted model
for i = 1:length(fittedModel)
	pimaImputed{i} = impute(fittedModel{i},pimaNan);
	mse(i) = sum(sum(pimaImputed{i} - pima).^2)/prod(size(pimaImputed));
end


fitMethodNames = {'fully observed', 'partially observed'};
feature = {'glu', 'bp', 'skin', 'bmi'};
% Plot the imputed vs. the true values, with a 45 degree line to indicate "perfect" imputation
for mod = 1:length(fittedModel)
  figure;
  for j = 1:p
    subplot(2,2,j); hold on;
    ndx = miss(:,j)==0;
    p1 = plot(pima(ndx,j), pimaImputed{mod}(ndx,j),'b+');
    %title(titles{j})
    xlabel(sprintf('true %s', feature{j}));
    ylabel(sprintf('imputed %s', feature{j}));
    axis equal
    V = axis; lowLim = max(V(1),V(3)); upLim = min(V(2),V(4));
    plot(lowLim:0.1:upLim, lowLim:0.1:upLim, 'r', 'linewidth', 3);
  end
  suptitle(sprintf('imputation using %s data', fitMethodNames{mod}));
end

if doPrintPmtk, printPmtkFigures('pima-realvsimputed'); end;

