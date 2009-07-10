nRow = 100;
nCol = 30;

%% First a really simple example
fprintf('First discover a bicluster in a really simple example -- binary data\n')
data = zeros(nRow, nCol);

data(5:5:end, 5:5:end) = 1;
patternRows = 5:5:size(data,1);
patternCols = 5:5:size(data,2);

%bicluster(data);
[dummyBcRow, dummyBcCol] = biclusterMultiple(data);
%% Now a more difficult example

fprintf('\n\nNow we try a more difficult example, similar to the example from Fig. 2 in Shen et al (2003): Biclustering Microarray Data in Gibbs Sampling\n')

clear data;
nLevels = 3;
data = unidrnd(nLevels, [nRow, nCol]);
% now we embed the pattern
patternRowSize = 25;
patternColSize = 8;
patternRows = randsample(nRow, patternRowSize);
patternCols = randsample(nCol, patternColSize);
fprintf('True rows: %s\n', mat2str(sort(patternRows')))
fprintf('True columns: %s', mat2str(sort(patternCols')))
sharp = 0.90;
p = [sharp, normalize(ones(1,nLevels-1))*(1-sharp)]; % a sharp multinomial distribution
p = perms(p);
p = p(unidrnd(size(p,1), patternColSize, 1),:);
for k=1:patternColSize
  data(patternRows,patternCols(k)) = sampleDiscrete(p(k,:), patternRowSize, 1);
end

%bicluster(data);
[simBcRow, simBcCol] = biclusterMultiple(data);

%{ Not working yet
fprintf('\n\nNow multiple biclusters\n')
clear data
nLevels = 3;
data = unidrnd(nLevels, [nRow, nCol]);
truecount = 3;
patternRowSizeVec = [25, 20, 10];
patternRowStart = [1, 30, 60];
patternColSizeVec = [8, 5, 2];
patternColStart = [1, 10, 18];

for j=1:truecount
  % now we embed the pattern
  patternRowSize = patternRowSizeVec(j);
  patternColSize = patternColSizeVec(j);

  patternRows = patternRowStart(j):(patternRowStart(j) + patternRowSize - 1);
  patternCols = patternColStart(j):(patternColStart(j) + patternColSize - 1);
  %fprintf('True rows: %s\n', mat2str(sort(patternRows')))
  %fprintf('True columns: %s', mat2str(sort(patternCols')))
%  p = [sharp, normalize(ones(1,nLevels-1))*(1-sharp)]; % a sharp multinomial distribution
%  p = perms(p);
%  p = p(unidrnd(size(p,1), patternColSize, 1),:);
  for k=1:patternColSize
    data(patternRows, patternColStart(j) + k) = sampleDiscrete(p(j,:), patternRowSize, 1);
  end
end

[multBcRow, multBcCol] = biclusterMultiple(data);
%}