

methods = {};
methods{end+1} = 'glassoCoordDesc';
methods{end+1} = 'glassoR';

% Timing on random problems 
d = 10;
setSeed(0);
ns = [d*2 d/2];
clear precMat
lambda = 0.1;
for trial=1:length(ns)
  n = ns(trial);
  X = randn(n,d);
  S = cov(X) + 0.0001*eye(d);
  fprintf('n=%d, d=%d, S is pd%d\n\n', n, d, isposdef(S));
  figure;
  for m=1:length(methods)
    tic;
    precMat{m} = feval(methods{m}, S, lambda);
    t=toc;
    correct = approxeq(precMat{m}, precMat{1}, 1e-1);
    fprintf('method %s correct %d, time %5.3f\n', ...
      methods{m}, correct, t);
    subplot(2,2,m); imagesc(precMat{m}); 
    title(sprintf('n%d, d%d, %s', n, d, methods{m}));
  end
end


