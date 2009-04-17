%% Demo of how to compute discrete approximaiton to 1d/2d Gaussian posterior 

mu = [5 1]'; Sigma = 2*eye(2);
Zexact2 = sqrt(det(2*pi*Sigma))

% picking the right range is crucial...
range1 = 0:0.1:10; range2 = -5:0.05:5;
 
target = @(X) gausspdfUnnormalized(X,mu,Sigma);
papprox2 = GridDist(target, range1, range2);
Zapprox2 = exp(lognormconst(papprox2))

%figure; plot(papprox2, 'type', 'heatmap');
%figure; plot(papprox2, 'type', 'contour');


%% Moments 

mu(:)'
approxMean2 = mean(papprox2)

for d=1:2
  papprox1{d} = marginal(papprox2, d);
  Zexact1(d) = sqrt(det(2*pi*Sigma(d,d)));
  %Zapprox1(d) = exp(lognormconst(papprox1{d}));
  approxMean1(d) = mean(papprox1{d});
  approxVar1(d) = var(papprox1{d});
end

fprintf('Mean\n');
for d=1:2
fprintf('%d &  %5.3f & %5.3f\n', d, mu(d), approxMean1(d));
end
fprintf('Var\n');
for d=1:2
fprintf('%d &  %5.3f & %5.3f\n', d, Sigma(d,d), approxVar1(d));
end
%{
fprintf('Z\n');
for d=1:2
fprintf('%d &  %5.3f & %5.3f\n', d, Zexact1(d), Zapprox1(d));
end
%}
