function GPSR_BB_path_timing_demo()

% Now do timing experiments
ntrials = 5;
n = 100; d= 1000; sigma = 1;
X = randn(n,d);
s = rand(d,1)>0.5;
w = rand(d,1) .* s;
y = X*w + sigma*randn(n,1);
time = zeros(2, ntrials);
names = {'gpsr', 'lars'};
for i=1:ntrials
  t = cputime;
  [w] = GPSR_BB_path(X, y);
  time(1,i) = cputime-t;
  
  t = cputime;
  [w] = lars(X, y, 'lasso');
  time(2,i) = cputime-t;
end
figure;
boxplot(time', 'labels', names);
title(sprintf('n=%d,d=%d',n,d));