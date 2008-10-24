function v = sentenceToWordCount(s, words)

tokens = regexp(s, '(\w+)', 'match');
v = zeros(length(words),1);
for j=1:length(tokens)
  k = strmatch(tokens{j}, words, 'exact');
  v(k) = v(k) + 1;
end
