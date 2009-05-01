%% Sequential Updating of Sigma in 1d given fixed mean
%#testPMTK
nu = 0.001; S = 0.001; 
setSeed(1);
mutrue = 5; Ctrue = 10;
mtrue = MvnDist(mutrue, Ctrue);
n = 500;
X = sample(mtrue, n);
ns = [0 2 5 50];
fig1= figure; hold on;
fig2 = figure;
pmax = -inf;
[styles, colors, symbols] =  plotColors();
for i=1:length(ns)
    prior = InvWishartDist(nu, S);
    n = ns(i);
    m = fit(Mvn_InvWishartDist(mutrue, prior), 'data', X(1:n));
    post = m.SigmaDist;
    mean(post);
    figure(fig1);
    [h(i), p]= plot(post, 'plotArgs', {styles{i}, 'linewidth', 2}, 'xrange', [0 15]);
    legendstr{i} = sprintf('n=%d', n);
    pmax = max(pmax, max(p));
    xbar = mean(X(1:n)); vbar = var(X(1:n));
    %hh(i)=line([vbar vbar], [0 pmax],'color',colors(i),'linewidth',3);

    if nu<1 && n==0, continue; end % improper prior, cannot sample from it
    figure(fig2); subplot(length(ns),1,i);
    XX = sample(post,100);
    hist(XX)
    title(legendstr{i})
    suptitle('samples from X');
end
figure(fig1);
legend(h,legendstr);
titlestr = sprintf('prior = IW(%s=%5.3f, S=%5.3f), true %s=%5.3f', ...
    '\nu', nu, S, '\sigma^2', Ctrue);
title(titlestr)
line([Ctrue Ctrue], [0 pmax],'color','k','linewidth',3);
if doPrintPmtk; printPmtkFigures('sigmaPost'); end;

figure(fig2); suptitle(titlestr);
