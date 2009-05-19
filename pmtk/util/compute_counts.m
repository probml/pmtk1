function count = compute_counts(data, sz)
% COMPUTE_COUNTS Count the number of times each combination of discrete assignments occurs
% count = compute_counts(data, sz)
%
% data(i,t) is the value of variable i in case t - should be {1, 2, ... K} 
% sz(i) : values for variable i are assumed to be in [1:sz(i)]
%
% Example: to compute a transition matrix for an HMM from a sequence of labeled states:
% transmat = mk_stochastic(compute_counts([seq(1:end-1); seq(2:end)], [nstates nstates]));
%
% Example: gs = [1 2 3 1]; os = [1 2 3 1]; D = [gs;os]; C=compute_counts(D,[ 3 3])
% C =
%     2     0     0
%     0     1     0
%     0     0     1
     
D = data'; % each row of data is a case
[ncases, nvars] = size(D); %#ok
assertKPM(length(sz) == nvars);
P = prod(sz);
indices = subv2ind(sz, D); % encode each row as an integer 
%count = histc(indices, 1:P);
count = hist(indices, 1:P); % count how many times each integer occurs
count = reshapePMTK(count, sz);

