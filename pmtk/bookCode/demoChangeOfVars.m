%#demo

xs = -1:0.1:1;
dist = UnifDist(-1,1);
px = exp(logprob(dist, xs));
fn = @(x) x.^2;
ys = fn(xs);

% analtyic
ppy = 1./(2*sqrt(ys));

% Monte Carlo
n=1000;
samples = sample(dist, 1000);
samples2 = fn(samples);
%[f, xi] = ksdensity(samples2);
[h,bins] = hist(samples2,20);
h = normalize(h);

figure(2);clf
nr = 1; nc = 3;
subplot(nr,nc,1); plot(xs, px, '-');
subplot(nr,nc,2); plot(ys, ppy, '-');
subplot(nr,nc,3);  bar(bins,h);
