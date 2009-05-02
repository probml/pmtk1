function [] = doPrintPmtkFigures(filename)
  printFolder = '/ubc/cs/home/n/nevek/fig';
  % create an eps and pdf file for the figure
  pdfcrop;
  exportfig(gcf, sprintf('%s/%s.eps', printFolder, filename), 'Resolution', 1200, 'Format', 'eps' );
  exportfig(gcf, sprintf('%s/%s.pdf', printFolder, filename), 'Resolution', 1200, 'Format', 'pdf' );
  %export_fig(fullfile(printFolder, sprintf('%s.pdf', filename)), '-pdf');
  %export_fig(fullfile(printFolder, sprintf('%s.eps', filename)), '-eps');
end
