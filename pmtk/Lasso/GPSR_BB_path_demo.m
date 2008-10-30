function GPSR_BB_path_demo()

% Reproduce fig 3.9  on p65 of "Elements of statistical learning" 

foo=load('prostate.mat'); % from prostateDataMake
X = center(foo.Xtrain); X = mkUnitVariance(X); y = center(foo.ytrain);
names = foo.names;

taumax = max(abs(X'*y)); % beyond this, w is all 0s
taus = linspace(0.9,0,20) * taumax;
[w, wdb] = GPSR_BB_path(X, y, taus, true);
doPlot(w); title('GPSR L1')
%doPlot(wdebiased); title('GPSR debiased')

wLars = lars(X, y, 'lasso');
doPlot(wLars); title('Lars')

lambdas = 2*taus; % gpsr uses min 1/2 RSS(w) + tau ||w||_1, lars uses RSS(w) + lambda ||w||_1
winterp = interpolateLarsWeights(wLars,lambdas,X,y);
figure; imagesc(w); title('GPSR'); colorbar
figure; imagesc(wLars); title('lars'); colorbar
figure; imagesc(winterp); title('lars interp'); colorbar


  function doPlot(w)
    wLS = X\y; denom = sum(abs(wLS'));
    s = sum(abs(w),2)/denom;
    figure;
    plot(s,w, 'o-')
    legend(names(1:8), 'location', 'northwest')
    title('LASSO path on prostate cancer data')
    xlabel(sprintf('shrinkage factor s(%s)', '\lambda'))
    set(gca,'xlim',[0 1])
  end

end