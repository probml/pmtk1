function printPmtkFiguresKPM(filename, printFolder)
% Save the current figure in multiple file formats
if nargin < 2
  printFolder = '/ubc/cs/home/n/nevek/fig';
end
pdfcrop;
opts = struct('Color', 'rgb', 'Resolution', 1200);
exportfig(gcf, sprintf('%s/%s.eps', printFolder, filename), opts, 'Format', 'eps' );
exportfig(gcf, sprintf('%s/%s.pdf', printFolder, filename), opts, 'Format', 'pdf' );
%export_fig(fullfile(printFolder, sprintf('%s.pdf', filename)), '-pdf');
%export_fig(fullfile(printFolder, sprintf('%s.eps', filename)), '-eps');
print(gcf, '-dpng', sprintf('%s/%s.png', printFolder, filename));
end
