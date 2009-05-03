%% Posterior of logistic regression params in 2d 
% Based on code written by Mark Girolami
setSeed(0);
% We generate data from two Gaussians:
% x|C=1 ~ gauss([1,5], I)
% x|C=0 ~ gauss([-5,1], 1.1I)
N=30;
d=2;
mu1=[ones(N,1) 5*ones(N,1)];
mu2=[-5*ones(N,1) 1*ones(N,1)];
class1_std = 1;
class2_std = 1.1;
X = [class1_std*randn(N,2)+mu1;2*class2_std*randn(N,2)+mu2];
y = [ones(N,1);zeros(N,1)];
D = DataTable(X, y+1); % make sure y is {1,2}
alpha=100; %Variance of prior (alpha=1/lambda)

%Limits and grid size for contour plotting
Range=8;
Step=0.1;
[w1,w2]=meshgrid(-Range:Step:Range,-Range:Step:Range);
[n,n]=size(w1);                                                     
W=[reshape(w1,n*n,1) reshape(w2,n*n,1)];

Range=12;
Step=0.1;
[x1,x2]=meshgrid(-Range:Step:Range,-Range:Step:Range);
[nx,nx]=size(x1);
grid=[reshape(x1,nx*nx,1) reshape(x2,nx*nx,1)];

% Plot data and plug-in predictive
figure;
m = fit(LogregBinaryL2('-lambda', 1/alpha), D);
[yhat, pred] = predict(m, grid);
p = pmf(pred); % pred is Bernoulli, p is a vector of numbers
contour(x1,x2,reshape(p,[nx,nx]),30);
hold on
plot(X((y==1),1),X((y==1),2),'r.');
plot(X((y==0),1),X((y==0),2),'bo');
title('p(y=1|x, wMAP)')
if doPrintPmtk, printPmtkFigures('logregLaplaceDemoGirolami-predPlugin'); end;

% Plot prior and posterior
eta=W*X';
Log_Prior = log(mvnpdf(W, zeros(1,d), eye(d).*alpha));
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
if doPrintPmtk, printPmtkFigures('logregLaplaceDemoGirolami-post'); end;

%Compute the Laplace Approximation

m = fit(LogregBinaryLaplace('-lambda', 1/alpha), D);
wMAP = m.wDist.mu(2:end); C = m.wDist.Sigma(2:end,2:end);
Log_Laplace_Posterior = log(mvnpdf(W, -wMAP', C)+eps);
subplot(J,K,4);
contour(w1,w2,reshape(-Log_Laplace_Posterior,[n,n]),30);
hold on
plot(W(j,1),W(j,2),'.','MarkerSize',40);
title('Laplace Approximation to Posterior')

% Posterior predictive
figure;
[yhat, pred] = predict(m,grid);
pmat = pmf(pred);
contour(x1,x2,reshape(pmat',[nx,nx]),30);
hold on
plot(X((y==1),1),X((y==1),2),'r.');
plot(X((y==0),1),X((y==0),2),'bo');
title('p(y=1|x, D)')

if doPrintPmtk, printPmtkFigures('logregLaplaceDemoGirolami-predLaplace'); end;

