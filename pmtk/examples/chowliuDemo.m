%% Find the MLE tree from a word-document binary matrix

load newsgroups % documents, wordlist, newsgroups
X = documents'; % 16,642 documents by 100 words  (sparse logical  matrix)

M = fit(DgmTreeTabular, 'data', X);
ll = logprob(M, X);
plotGraph(M, 'nodeLabels', wordlist)

% Plot loglikelihood of training cases
figure;hist(ll,100); title('log-likelihood of training cases using ChowLiu tree')

% Find words in datacases with best  and worst  likelihoods
[junk, ndx] = sort(ll, 'descend');
chosen = [ndx(1:5)' ndx(end-2:end-1)'];
for i=1:length(chosen)
  j = chosen(i);
  fprintf('words in sentence %d with loglik %5.3f\n', j, ll(j));
  wordlist(X(j,:))
end
