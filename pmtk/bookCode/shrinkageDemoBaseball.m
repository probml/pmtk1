% shrinkageDemoBaseball

shrinkageDemoBaseballData() % load data
Y = data(:,1);
p = data(:,2);
n = 45;
x = sqrt(n)*asin(2*Y-1); % arcsin transform
d = length(x);
xbar = mean(x);
V = sum((x-xbar).^2);
B = (d-3)/V;
shrunkTransformed = xbar + (1-B)*(x-xbar); % Efron-Morris shrinkage
shrunk = 0.5*(sin(shrunkTransformed/sqrt(n))+1); % untransform
MLE = Y;

figure(1); clf
plot(MLE, ones(1,d) ,'o');
hold on
plot(shrunk, 0*ones(1,d), 'o');
for i=1:d
  line([Y(i); shrunk(i)], [1; 0]);
end
title('MLE (top) and shrinkage estimates (bottom)')
if doPrintPmtk, printPmtkFigures('shrinkageDemoBaseballParams'); end;

figure(2);clf;bar([p'; MLE'; shrunk']')
legend('true', 'MLE', 'shrunk')
title(sprintf('MSE MLE = %6.4f, MSE shrunk = %6.4f', mse(p,MLE), mse(p,shrunk)));
if doPrintPmtk, printPmtkFigures('shrinkageDemoBaseballPred'); end;