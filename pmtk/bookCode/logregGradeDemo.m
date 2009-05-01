%function logregGradeDemo();
% logistic regression example on grade data from Johnson and Albert p87

stat = load('stat2.dat'); % Johnson and Albert p77 table 3.1
% stat=[pass(0/1), 1, 1, sat_score, grade in prereq]
% where the grade in prereq is encoded as A=5,B=4,C=3,D=2,F=1
y = stat(:,1);
N = length(y);
SAT = stat(:,4);

figure(3);clf
[junk,perm] = sort(SAT,'ascend');
N = length(perm);
for ii=1:N
  i = perm(ii);
  hold on
  h=plot(SAT(i), y(i), 'ko');
  set(h,'markerfacecolor', 'k');
end
title('data')
xlabel('SAT')
ylabel('pass/fail')



% log reg
X = [ones(N,1) SAT];
lambda = eps;
[beta, C, nll] = logregFitFminunc(y, X, lambda);

figure(1);clf
[junk,perm] = sort(SAT,'ascend');
N = length(perm);
for ii=1:N
  i = perm(ii);
  p = sigmoid(X(i,:)*beta); %1/(1+exp(-X(i,:)*beta));
  yhat = (p>0.5);
  if yhat==y(i)
    plot(SAT(i), p, 'gs');
  else
    plot(SAT(i), p, 'rx', 'markersize', 12);
  end
  hold on
  h=plot(SAT(i), y(i), 'ko');
  set(h,'markerfacecolor', 'k');
end
title('logistic regression')
if doPrintPmtk, printPmtkFigures('logregGradeLogreg'); end;



% draw the decision boundary
NN = 100;
SATdense = linspace(min(SAT),max(SAT),NN);
Xdense = [ones(NN,1) SATdense(:)];
ps = sigmoid(Xdense*beta);
ndx = find(ps > 0.5);
ndx = ndx(1);
line([SATdense(ndx) SATdense(ndx)], [0 1]);

% LS
beta = X\y;
[junk,perm] = sort(SAT,'ascend');
N = length(perm);

wrong = [];
figure(2);clf
for ii=1:N
  i = perm(ii);
  p = X(i,:)*beta;
  yhat = (p>0.5);
  if yhat==y(i)
    plot(SAT(i), p, 'gs');
  else
    wrong = [wrong i]
    plot(SAT(i), p, 'rx', 'markersize', 12);
  end
  hold on
  h=plot(SAT(i), y(i), 'ko');
  set(h,'markerfacecolor', 'k');
end
title('linear regression')
if doPrintPmtk, printPmtkFigures('logregGradeLinreg'); end;

% draw the decision boundary
ps = Xdense*beta;
ndx = find(ps > 0.5);
ndx = ndx(1);
line([SATdense(ndx) SATdense(ndx)], [0 1]);
