%% Logreg Plot Demo
% Based on code written by Mark Girolami
setSeed(0);
% We generate data from two Gaussians:
% x|C=1 ~ gauss([1,5], I)
% x|C=0 ~ gauss([-5,1], 1.1I)
N=30;
D=2;
mu1=[ones(N,1) 5*ones(N,1)];
mu2=[-5*ones(N,1) 1*ones(N,1)];
class1_std = 1;
class2_std = 1.1;
X = [class1_std*randn(N,2)+mu1;2*class2_std*randn(N,2)+mu2];
y = [ones(N,1);zeros(N,1)];
alpha=100; %Variance of prior (alpha=1/lambda)

%Limits and grid size for contour plotting
Range=8;
Step=0.1;
[w1,w2]=meshgrid(-Range:Step:Range,-Range:Step:Range);
[n,n]=size(w1);                                                      %#ok
W=[reshape(w1,n*n,1) reshape(w2,n*n,1)];

Range=12;
Step=0.1;
[x1,x2]=meshgrid(-Range:Step:Range,-Range:Step:Range);
[nx,nx]=size(x1);
grid=[reshape(x1,nx*nx,1) reshape(x2,nx*nx,1)];

% Plot data and plug-in predictive
figure;
m = fit(LogregDist, 'X', X, 'y', y+1);
pred = predict(m,grid);

contour(x1,x2,reshape(pred.mu(2,:)',[nx,nx]),30);
hold on
plot(X((y==1),1),X((y==1),2),'r.');
plot(X((y==0),1),X((y==0),2),'bo');

title('p(y=1|x, wMLE)')

% Plot prior and posterior
eta=W*X';
Log_Prior = log(mvnpdf(W, zeros(1,D), eye(D).*alpha));
Log_Like = eta*y - sum(log(1+exp(eta)),2);
Log_Joint = Log_Like + Log_Prior;
figure;
J=2;K=2;
subplot(J,K,1)
contour(w1,w2,reshape(-Log_Prior,[n,n]),30);
title('Log-Prior');
subplot(J,K,2)
contour(w1,w2,reshape(-Log_Like,[n,n]),30);
title('Log-Likelihood');
subplot(J,K,3)
contour(w1,w2,reshape(-Log_Joint,[n,n]),30);
title('Log-Unnormalised Posterior')
hold on

%Identify the parameters w1 & w2 which maximize the posterior (joint)
[i,j]=max(Log_Joint);                                                               
plot(W(j,1),W(j,2),'.','MarkerSize',40);
%Compute the Laplace Approximation
tic
m = fit(Logreg_MvnDist(),'X',X,'y',y+1,'priorStrength',1/alpha,'infMethod','laplace');
toc
wMAP = m.wDist.mu;
C = m.wDist.Sigma;
Log_Laplace_Posterior = log(mvnpdf(W, -wMAP', C)+eps);
subplot(J,K,4);
contour(w1,w2,reshape(-Log_Laplace_Posterior,[n,n]),30);
hold
plot(W(j,1),W(j,2),'.','MarkerSize',40);
title('Laplace Approximation to Posterior')
% Posterior predictive
% wMAP
figure;
subplot(2,2,1)
pred = predict(m,grid);

contour(x1,x2,reshape(pred.mu(2,:)',[nx,nx]),30);
hold on
plot(X((y==1),1),X((y==1),2),'r.');
plot(X((y==0),1),X((y==0),2),'bo');

title('p(y=1|x, wMAP)')
subplot(2,2,2); hold on
S = 100;
plot(X((y==1),1),X((y==1),2),'r.');
plot(X((y==0),1),X((y==0),2),'bo');
predDist = predict(m,grid,'method','mc','nsamples',S);


% for s=1:min(S,20)
%     p = samples(s,:);
%     contour(x1,x2,reshape(p,[nx,nx]),[0.5 0.5]);
% end
% set(gca, 'xlim', [-10 10]);
% set(gca, 'ylim', [-10 10]);
% title('decision boundary for sampled w')
% subplot(2,2,3)

meanPred = mean(pred)';
contour(x1,x2,reshape(meanPred(:,1),[nx,nx]),30);
hold on
plot(X((y==1),1),X((y==1),2),'r.');
plot(X((y==0),1),X((y==0),2),'bo');


title('MC approx of p(y=1|x)')
subplot(2,2,4)
pred = predict(m,grid,'method','integral');

contour(x1,x2,reshape(pred.mu(2,:)',[nx,nx]),30);
hold on
plot(X((y==1),1),X((y==1),2),'r.');
plot(X((y==0),1),X((y==0),2),'bo');

title('numerical approx of p(y=1|x)')
