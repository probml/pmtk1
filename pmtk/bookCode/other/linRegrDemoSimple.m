%function linRegrDemoSimple()

seed = 1; randn('state', seed); rand('state', seed);

xTrainRaw = (-10:1:10)';
Ntrain = length(xTrainRaw);
xTrain = [ones(Ntrain,1) xTrainRaw(:)];
%xTrain = [xTrainRaw(:)];
yTrain = 2.*xTrainRaw+3 + 5*randn(Ntrain,1);

X = xTrain;
y = yTrain;
w = pinv(X'*X)*X'*y; % OLS estimate
%w = X\y;
yPredTrain = xTrain*w;


% Performance on test set

%xTestRaw = (-9.5:1:9.5)';
xTestRaw = (-10.5:1:10.5)';
Ntest = length(xTestRaw);
xTest = [ones(Ntest,1) xTestRaw(:)];
%xTest = [xTestRaw(:)];
yTestOpt = 2.*xTestRaw+3;
yPredTest = xTest*w;


figure(2); clf
hold off
h=plot(xTestRaw, yPredTest,  'k-');set(h, 'linewidth', 2)
hold on
%h=plot(xTestRaw,yTestOpt,'bx-');set(h, 'linewidth', 2)
h = plot(xTrainRaw, yTrain, 'ro');set(h, 'linewidth', 2)
grid on

h = plot(xTrainRaw, yPredTrain, 'bx', 'markersize', 10);set(h, 'linewidth', 2)
for i=1:Ntrain
  h=line([xTrainRaw(i) xTrainRaw(i)], [yPredTrain(i) yTrain(i)]);
  set(h, 'linewidth', 2, 'color', 'b')
end

%legend('prediction on test', 'true (noise free) test', ...
%       'training fit', 'training data','location','northwest')

