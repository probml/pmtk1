load statlog.heart.data

X = standardizeCols(statlog(:,1:end-1));
y = statlog(:,end);

[n,p] = size(X);

%% Solve with minFunc

Xmf = [ones(n,1) X];
ymf = sign(y-1.5);
options.display = 'none';
options.Method = 'newton';
tic
w = minFunc(@LogisticLoss,zeros(p+1,1),options,Xmf,ymf);
toc

%% Solve with glmfit

tic
b = glmfit(X,y-1,'binomial','link','logit');
toc

%% Solve with IRLS

tic
w2 = L2LogReg_IRLS(Xmf,ymf);
toc