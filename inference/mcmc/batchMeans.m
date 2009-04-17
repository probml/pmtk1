function [meanB, se, acf] = batchMeans(x, batchsize)
% meanB(b,i) = mean of x(s,i) for s in batch b

[nsamples nchains] = size(x);
nbatches = floor(nsamples/batchsize);
meanB = zeros(nbatches, nchains);
for b=1:nbatches
  if b==nbatches
    xs = x((b-1)*batchsize + 1: end, :);
  else
    xs = x((b-1)*batchsize + 1: b*batchsize, :);
  end
  meanB(b,:) = mean(xs,1);
end
se = std(meanB,1)/sqrt(nbatches);
acf = acorr(meanB,1);
