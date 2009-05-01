%% Plot a spectogram and its MFCC representation
% Data source: Tommi Jaakkola
load data45;
figure; specgram(signal1); title('spectogram of "four"');
if doPrintPmtk, printPmtkFigures('speechFourSpectogram'); end;
figure; imagesc(train4{1}); title('mfcc of "four"');
if doPrintPmtk, printPmtkFigures('speechFourMFCC2'); end;