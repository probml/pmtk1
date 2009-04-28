%% All subsets regression on prostate cancer data
% Reproduce fig 3.5  on p56 of Elements 1st ed

%#slow

clear all
load('prostate.mat') % from prostateDataMake
ndxTrain = find(istrain);
ytrain = y(ndxTrain); Xtrain = X(ndxTrain,:);
Dtrain = DataTable(Xtrain, ytrain, names);

T = StandardizeTransformer(false);
ML = LinregAllSubsetsModelList('-transformer', T, '-verbose', true);
ML = fit(ML, Dtrain);
Nm = length(ML.models);
for m=1:Nm
  rss(m) = sum(squaredErr(ML.models{m}, Dtrain));
  sz(m) = sum(abs(ML.models{m}.w) ~= 0);
end
figure;
plot(sz, rss, '.');
hold on
% Lower envelope
d = ndimensions(Dtrain);
for i=0:d
  ndx = find(sz==i);
  [bestScore(i+1) bestSet] = min(rss(ndx));
end
plot(0:d, bestScore, 'ro-')
xlabel('subset size')
ylabel('RSS on training set')
title('all subsets on prostate cancer')
