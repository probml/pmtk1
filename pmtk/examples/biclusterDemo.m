%cls; 
setSeed(0);
nRow = 100;
nCol = 30;
nLevels = 3;


%% First a really simple example
fprintf('First discover a bicluster in a really simple example -- binary data\n')
data = zeros(nRow, nCol);



data(5:5:end, 5:5:end) = 1;
patternRows = 5:5:size(data,1);
patternCols = 5:5:size(data,2);

%bicluster(data);
setSeed(1);
[dummyBcRow, dummyBcCol, traceDummyRow, traceDummyCol] = biclusterMultiple(data, '-trace', true);
for c=1:size(traceDummyRow)
  figure(); plot(traceDummyRow{c}'); legend();
  figure(); plot(traceDummyCol{c}'); legend();
end
if(doPrintPmtk)
  figure(1);
  printPmtkFiguresKPM('dummyBiclusterData'); 
  figure(2);
  printPmtkFiguresKPM('dummyTraceRows');
  figure(3);
  printPmtkFiguresKPM('dummyTraceCols');
end;

% Produce the numbers for a contingency table (hint: add length() command around each)
% intersect(patternRows, uniBcRow{1});
% setdiff(patternRows, uniBcRow{1});
% intersect(setdiff(1:nRow, patternRows), uniBcRow{1})
% intersect(setdiff(1:nRow, patternRows), setdiff(1:nRow, uniBcRow{1}))

% intersect(patternCols, uniBcCol{1});
% setdiff(patternCols, uniBcCol{1});
% intersect(setdiff(1:nCol, patternCols), uniBcCol{1})
% intersect(setdiff(1:nCol, patternCols), setdiff(1:nCol, uniBcCol{1}))


%% Now a more difficult example
fprintf('\n\nNow we try a more difficult example, similar to the example from Fig. 2 in Shen et al (2003): Biclustering Microarray Data in Gibbs Sampling\n')

clear data;
data = unidrnd(nLevels, [nRow, nCol]);
% now we embed the pattern
patternRowSize = 25;
patternColSize = 8;
patternRows = randsample(nRow, patternRowSize);
patternCols = randsample(nCol, patternColSize);
sharp = 0.90;
nLevels = 3;
p = [sharp, normalize(ones(1,nLevels-1))*(1-sharp)]; % a sharp multinomial distribution
p = perms(p);
p = p(unidrnd(size(p,1), patternColSize, 1),:);

fprintf('True rows: %s\n', mat2str(sort(patternRows')))
fprintf('True columns: %s\n', mat2str(sort(patternCols')))
for k=1:patternColSize
  data(patternRows,patternCols(k)) = sampleDiscrete(p(k,:), patternRowSize, 1);
end

%bicluster(data);
setSeed(2);
[uniBcRow, uniBcCol] = biclusterMultiple(data);
if(doPrintPmtk)
  figure(1);
  printPmtkFiguresKPM('uniBiclusterData'); 
end;

fprintf('\n\nNow multiple biclusters on a larger data matrix\n')
clear data patternRows patternCols
nRow = 200;
nCol = 40;
sharp = 0.90;
p = [sharp, normalize(ones(1,nLevels-1))*(1-sharp)]; % a sharp multinomial distribution
p = perms(p);
p = p(unidrnd(size(p,1), patternColSize, 1),:);
data = unidrnd(nLevels, [nRow, nCol]);
truecount = 3;
patternRowSizeVec = [40, 25, 35];
%patternRowStart = [1, 30, 60];
patternColSizeVec = [7, 2, 8];
%patternColStart = [1, 10, 18];

%rowProb = normalize(ones(1, nRow));
%colProb = normalize(ones(1, nRow));
%patternRowsVec = sample(rowProb, sum(patternRowSizeVec));
%patternColsVec = sample(colProb, sum(patternColSizeVec));
for j=1:truecount
  % now we embed the pattern
  patternRowSize = patternRowSizeVec(j);
  patternColSize = patternColSizeVec(j);

  patternRows{j} = randsample(nRow, patternRowSize);
  patternCols{j} = randsample(nCol, patternColSize);
  fprintf('True rows: %s\n', mat2str(sort(patternRows{j}')))
  fprintf('True columns: %s\n', mat2str(sort(patternCols{j}')))
%  patternRows = patternRowStart(j):(patternRowStart(j) + patternRowSize - 1);
%  patternCols = patternColStart(j):(patternColStart(j) + patternColSize - 1);
  for k=1:patternColSize
    data(patternRows{j}, patternCols{j}(k)) = sampleDiscrete(p(j,:), patternRowSize, 1);
  end
end

setSeed(3);
[multBcRow, multBcCol] = biclusterMultiple(data);
