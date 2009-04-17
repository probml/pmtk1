% Reproduce fig 3.7  on p61 of "Elements of statistical learning" 

clear all
load('prostate.mat') % from prostateDataMake
lambdas = [logspace(4, 0, 20) 0];
ndx = find(istrain);
y = y(ndx); X = X(ndx,:);

[w, mseTrain, mseTest, df, gcv] = ridgeSVD(X, y,  [], [], lambdas);
figure(1);clf
plot(df, w(2:end,:)', 'o-')
legend(names(1:8))


Xc = standardize(X);
yc = center(y);
[w2, df2] = ridgePathSimple(Xc, yc, lambdas);
figure(2);clf
plot(df2, w2, 'o-')
legend(names(1:8))
