function [] = printPmtkFigures(filename)
  printFolder = '/ubc/cs/home/n/nevek/fig';
  % create an eps and pdf file for the figure
  pdfcrop;
  opts = struct('Color', 'rgb', 'Resolution', 1200);
  exportfig(gcf, sprintf('%s/%s.eps', printFolder, filename), opts, 'Format', 'eps' );
  exportfig(gcf, sprintf('%s/%s.pdf', printFolder, filename), opts, 'Format', 'pdf' );
  %export_fig(fullfile(printFolder, sprintf('%s.pdf', filename)), '-pdf');
  %export_fig(fullfile(printFolder, sprintf('%s.eps', filename)), '-eps');
end
