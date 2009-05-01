load('yeastData310.mat') % 'X', 'genes', 'times');


figure;imagesc(X);colormap(redgreencmap)
xlabel('time')
set(gca,'xticklabel',times)
ylabel('genes')
title('yeast microarray data')
colorbar
if doPrintPmtk, pdfcrop; doPrintPmtkFigures('yeastHeatMap'); end;

%figure; plot(X'); set(gca,'xticklabel',times);
figure; plot(times,X,'o-');
xlabel('time')
set(gca,'xticklabel',times)
set(gca,'xtick',times)
ylabel('genes')
title('yeast microarray data')
set(gca,'xlim',[0 max(times)])
if doPrintPmtk, doPrintPmtkFigures('yeastTimeSeries'); end;
