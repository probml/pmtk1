%PRINT2ARRAY  Exports a figure to an image array
%#author Oliver Woodford
%#url http://www.mathworks.com/matlabcentral/fileexchange/23629
%
% Examples:
%   A = print2array
%   A = print2array(figure_handle)
%   A = print2array(figure_handle, resolution)
%
% This function outputs a bitmap image of the given figure, at the desired
% resolution.
%
% IN:
%   figure_handle - The handle of the figure to be exported. Default: gcf.
%   resolution - Resolution of the output, as a factor of screen
%                resolution. Default: 1.
%
% OUT:
%   A - MxNx3 uint8 image of the figure.

% Copyright (C) Oliver Woodford 2008-2009

function A = print2array(fig, res)
% Generate default input arguments, if needed
if nargin < 2
    res = 1;
    if nargin < 1
        fig = gcf;
    end
end
% Generate temporary file name
tmp_nam = [tempname '.tif'];
% Set paper size
old_mode = get(fig, 'PaperPositionMode');
set(fig, 'PaperPositionMode', 'auto');
% Print to tiff file
print(fig, '-opengl', ['-r' num2str(get(0, 'ScreenPixelsPerInch')*res)], '-dtiff', tmp_nam);
% Reset paper size
set(fig, 'PaperPositionMode', old_mode);
% Read in the printed file
A = imread(tmp_nam);
% Delete the file
delete(tmp_nam);
return
