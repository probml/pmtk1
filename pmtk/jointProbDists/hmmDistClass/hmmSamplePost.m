function [samples] = hmmSamplePost(initDist, transmat, obslik, nsamples)
% Forwards filtering, backwards sampling for HMMs
% OUTPUT:
% samples(t,s) = value of S(t)  in sample s

[K T] = size(obslik);
alpha = hmmFilter(initDist, transmat, obslik);
samples = zeros(T, nsamples);
dist = normalize(alpha(:,T));
samples(T,:) = sample(dist, nsamples);
for t=T-1:-1:1
  tmp = obslik(:,t+1) ./ (alpha(:,t+1)+eps); % b_{t+1}(j) / alpha_{t+1}(j)
  xi_filtered = transmat .* (alpha(:,t) * tmp');
  for n=1:nsamples
    dist = xi_filtered(:,samples(t+1,n));
    samples(t,n) = sample(dist);
  end
end


