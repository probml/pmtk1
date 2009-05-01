function [] = printPmtkFigures(filename)
  printFolder = '/ubc/cs/home/n/nevek/fig';
  % create an eps and pdf file for the figure
  pdfcrop;
  export_fig(fullfile(printFolder, sprintf('%s.pdf', filename)), '-pdf');
  export_fig(fullfile(printFolder, sprintf('%s.pdf', filename)), '-eps');
end
