%PRINT_PDF  Prints cropped figures or axes to pdf with fonts embedded
%
% Examples:
%   print_pdf filename
%   print_pdf(filename, handle)
%
% This function saves a figure or set of axes as a pdf nicely, without the
% need to specify multiple options. It improves on MATLAB's print command
% (using default options) in several ways:
%   - The figure borders are cropped
%   - Fonts are embedded (as subsets)
%   - Lossless compression is used on vector graphics
%   - High quality jpeg compression is used on bitmaps
%   - Dotted/dashed line dash lengths vary with line width (as on screen)
%   - Grid lines given their own dot style, instead of dashed
%
% This function requires that you have ghostscript installed on your system
% and that the executable binary is on your system's path. Ghostscript can
% be downloaded from: http://www.ghostscript.com
%
%IN:
%   filename - string containing the name (optionally including full or
%              relative path) of the file the figure is to be saved as. A
%              ".pdf" extension is added if not there already. If a path is
%              not specified, the figure is saved in the current directory. 
%   handle - The handle of the figure or axes to be saved. Default: current
%            figure.
%
% Copyright (C) Oliver Woodford 2008

% This function is inspired by Peder Axensten's SAVEFIG (fex id: 10889)
% which is itself inspired by EPS2PDF (fex id: 5782)
% The idea of editing the EPS file to change line styles comes from Jiro
% Doke's FIXPSLINESTYLE (fex id: 17928)
% The idea of changing dash length with line width came from comments on
% fex id: 5743, but the implementation is mine :)

% $Id: eps2pdf.m,v 1.1 2009/03/25 14:11:54 ojw Exp $

function eps2pdf(source, dest)
% Construct the options string for ghostscript
options = ['-q -dNOPAUSE -dBATCH -dEPSCrop -sDEVICE=pdfwrite -dPDFSETTINGS=/prepress -sOutputFile="' dest '" -f "' source '"'];
% Convert to pdf using ghostscript
%ghostscript(options);
system(sprintf('gs %s', options));
return

