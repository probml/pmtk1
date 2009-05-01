%EXPORT_FIG  Exports figures suitable for publication
%#author Oliver Woodford
%#url http://www.mathworks.com/matlabcentral/fileexchange/23629
%
% Examples:
%   export_fig filename
%   export_fig filename -format1 -format2
%   export_fig(..., handle)
%
% This function saves a figure or single axes to one or more vector and/or
% bitmap formats, with the following properties:
%   - Figure/axes reproduced as it appears on screen
%   - Cropped borders
%   - Embedded fonts (vector formats)
%   - Improved line and grid line styles (vector formats)
%   - Anti-aliased graphics (bitmap formats)
%   - Transparent background supported
%   - Vector formats: pdf, eps
%   - Bitmap formats: png (supports transparency), tif, jpg, bmp 
%   - No options to be set
%   
% This function is especially suited to exporting figures for use in
% publications and presentations, because of the high quality and
% portability of media produced.
%
% Note that the background color and figure dimensions are reproduced
% (the latter approximately, and ignoring cropping) in the output file. For
% transparent background, set the figure (and axes) 'Color' property to
% 'none'; pdf, eps and png are the only formats to support transparency.
%
% When exporting to vector format (pdf & eps), this function requires that
% ghostscript is installed on your system. You can download this from:
%   http://www.ghostscript.com
% When exporting to eps it additionally requires pdftops, from the Xpdf
% suite of functions. You can download this from:
%   http://www.foolabs.com/xpdf
%
%IN:
%   filename - string containing the name (optionally including full or
%              relative path) of the file the figure is to be saved as. If
%              a path is not specified, the figure is saved in the current
%              directory. If neither a file extension nor a format are
%              specified, a ".png" is added and the figure saved in that
%              format.
%   -format1, -format2, etc. - strings containing the extensions of the
%                              file formats the figure is to be saved as.
%                              Valid options are: '-pdf', '-eps', '-png',
%                              '-tif', '-jpg' and '-bmp'. All combinations
%                              of formats are valid.
%   handle - The handle of the figure or axes to be saved. Default: gcf.

% Copyright (C) Oliver Woodford 2008-2009

% The idea of using ghostscript is inspired by Peder Axensten's SAVEFIG
% (fex id: 10889) which is itself inspired by EPS2PDF (fex id: 5782).
% The idea for using pdftops came from the MATLAB newsgroup (id: 168171).
% The idea of editing the EPS file to change line styles comes from Jiro
% Doke's FIXPSLINESTYLE (fex id: 17928).
% The idea of changing dash length with line width came from comments on
% fex id: 5743, but the implementation is mine :)
% The idea of anti-aliasing bitmaps came from Anders Brun's MYAA (fex id:
% 20979).

% $Id: export_fig.m,v 1.7 2009/04/11 16:16:25 ojw Exp $

function export_fig(varargin)
% Parse the input arguments
[name fig formats] = parse_args(varargin{:});
% Isolate the subplot, if it is one
cls = false;
if strcmp(get(fig, 'Type'), 'axes')
    % Given a handle of a single set of axes
    fig = isolate_subplot(fig);
    cls = true;
else
    old_mode = get(gcf, 'InvertHardcopy');
end
% Added by Cody - this will allow us to get a transparent background
set(fig, 'Color', 'none');
% Set to print exactly what is there
set(fig, 'InvertHardcopy', 'off');
% Do the vector formats first
if isvector(formats)
    % Generate an eps
    tmp_nam = [tempname '.eps'];
    print2eps(tmp_nam, fig);
    % Generate a pdf
    if formats.pdf
        pdf_nam = [name '.pdf'];
    else
        pdf_nam = [tempname '.pdf'];
    end
    eps2pdf(tmp_nam, pdf_nam);
    % Delete the eps
    delete(tmp_nam);
    if formats.eps
        % Generate an eps from the pdf
        pdf2eps(pdf_nam, [name '.eps']);
        if ~formats.pdf
            % Delete the pdf
            delete(pdf_nam);
        end
    end
end
% Now do the bitmap formats
if isbitmap(formats)
    tcol = get(gcf, 'Color');
    if isequal(tcol, 'none') && formats.png
        % Get out an alpha channel
        % Set the background colour to something rare
        tcol = 255 - [13 3 17];
        set(fig, 'Color', tcol / 255);
        % Print large version to array
        A = print2array(fig, 4);
        % Set the background colour back to normal
        set(fig, 'Color', 'none');
        % Extract transparent pixels and crop the background
        [A alpha] = crop_background(A, tcol);
        % Set background pixels which will have non-zero alpha to the nearest
        % foreground colour, and paint others white
        A = inpaint_background(A, alpha);
        % Downscale the alphamatte
        alpha = quarter_size(single(alpha), 0);
        % Downscale the image
        A = quarter_size(A, 255);
        % Save the png
        imwrite(A, [name '.png'], 'Alpha', alpha);
        % Clear the png bit
        formats.png = false;
        % Get the non-alpha image
        if isbitmap(formats)
            alpha = repmat(alpha, [1 1 size(A, 3)]);
            A = uint8(single(A) .* alpha + 255 * (1 - alpha));
        end
        clear alpha
    else
        % Print large version to array
        if isequal(tcol, 'none')
            set(fig, 'Color', 'w');
            A = print2array(fig, 4);
            set(fig, 'Color', 'none');
            tcol = [255 255 255];
        else
            A = print2array(fig, 4);
            tcol = squeeze(A(1,1,:));
        end
        % Crop the background
        A = crop_background(A, tcol);
        % Downscale the image
        A = quarter_size(A, tcol);
    end
    % Save the images
    for a = {'png', 'tif', 'jpg', 'bmp'}
        if formats.(a{1})
            imwrite(A, [name '.' a{1}]);
        end
    end
end
if cls
    % Close the created figure
    close(fig);
else
    % Reset the hardcopy mode
    set(fig, 'InvertHardcopy', old_mode);
end
return

function [name fig formats] = parse_args(varargin)
% Parse the input arguments
% Get the name
name = varargin{1};
% Set the defaults
formats = struct('pdf', false, 'eps', false, 'png', false, 'tif', false, 'jpg', false, 'bmp', false);
if numel(name) > 3 && name(end-3) == '.' && any(strcmpi(name(end-2:end), {'pdf', 'eps', 'png', 'tif', 'jpg', 'bmp'}))
    formats.(lower(name(end-2:end))) = true;
    name = name(1:end-4);
end
fig = get(0, 'CurrentFigure');

% Go through the other arguments
for a = 2:nargin
    if ishandle(varargin{a})
        fig = varargin{a};
    elseif ischar(varargin{a}) && numel(varargin{a}) == 4 && varargin{a}(1) == '-' && any(strcmpi(varargin{a}(2:4), {'pdf', 'eps', 'png', 'tif', 'jpg', 'bmp'}))
        formats.(lower(varargin{a}(2:4))) = true;
    end
end

% Set the default format
if ~isvector(formats) && ~isbitmap(formats)
    formats.png = true;
end
return

function fh = isolate_subplot(ah, vis)
% Isolate the axes in a figure on their own
% Tag the axes so we can find them in the copy
old_tag = get(ah, 'Tag');
set(ah, 'Tag', 'AxesToCopy');
% Create a new figure exactly the same as the old one
fh = copyfig(get(ah, 'Parent')); %copyobj(get(ah, 'Parent'), 0);
if nargin < 2 || ~vis
    set(fh, 'Visible', 'off');
end
% Reset the axes tag
set(ah, 'Tag', old_tag);
% Get all the axes
axs = findobj(get(fh, 'Children'), 'Type', 'axes');
% Find the axes to save
ah = findobj(axs, 'Tag', 'AxesToCopy');
if numel(ah) ~= 1
    close(fh);
    error('Too many axes found');
end
I = true(size(axs));
I(axs==ah) = false;
% Set the axes tag to what it should be
set(ah, 'Tag', old_tag);
% Keep any legends which overlap the subplot
ax_pos = get(ah, 'OuterPosition');
ax_pos(3:4) = ax_pos(3:4) + ax_pos(1:2);
for ah = findobj(axs, 'Tag', 'legend')'
    leg_pos = get(ah, 'OuterPosition');
    leg_pos(3:4) = leg_pos(3:4) + leg_pos(1:2);
    % Overlap test
    if leg_pos(1) < ax_pos(3) && leg_pos(2) < ax_pos(4) &&...
       leg_pos(3) > ax_pos(1) && leg_pos(4) > ax_pos(2)
        I(axs==ah) = false;
    end
end
% Delete all axes except for the input axes and associated items
delete(axs(I));
return

function fh = copyfig(fh)
% Is there a legend?
if numel(findobj(get(fh, 'Children'), 'Type', 'axes', 'Tag', 'legend'))
    % copyobj will change the figure, so save and then load it instead
    tmp_nam = [tempname '.fig'];
    hgsave(fh, tmp_nam);
    fh = hgload(tmp_nam);
    delete(tmp_nam);
else
    % Safe to copy using copyobj
    fh = copyobj(fh, 0);
end
return

function A = quarter_size(A, padval)
% Downsample an image by a factor of 4
try
    % Faster, but requires image processing toolbox
    A = imresize(A, 1/4, 'bilinear');
catch
    % No image processing toolbox - resize manually
    % Lowpass filter - use Gaussian (sigma: 1.7) as is separable, so faster
    filt = single([0.0148395 0.0498173 0.118323 0.198829 0.236384 0.198829 0.118323 0.0498173 0.0148395]);
    if numel(padval) == 3 && padval(1) == padval(2) && padval(2) == padval(3)
        padval = padval(1);
    end
    if numel(padval) == 1
        B = repmat(single(padval), [size(A, 1) size(A, 2)] + 8);
    end
    for a = 1:size(A, 3)
        if numel(padval) == 3
            B = repmat(single(padval(a)), [size(A, 1) size(A, 2)] + 8);
        end
        B(5:end-4,5:end-4) = A(:,:,a);
        A(:,:,a) = conv2(filt, filt', B, 'valid');
    end
    clear B
    % Subsample
    A = A(2:4:end,2:4:end,:);
end
% Check if the image is greyscale
if size(A, 3) == 3 && ...
        all(reshape(A(:,:,1) == A(:,:,2), [], 1)) && ...
        all(reshape(A(:,:,2) == A(:,:,3), [], 1))
    A = A(:,:,1); % Save only one channel for 8-bit output
end
return

function [A alpha] = crop_background(A, bcol)
% Map the foreground pixels
alpha = A(:,:,1) ~= bcol(1) | A(:,:,2) ~= bcol(2) | A(:,:,3) ~= bcol(3);
% Crop the background
N = any(alpha, 1);
M = any(alpha, 2);
M = find(M, 1):find(M, 1, 'last');
N = find(N, 1):find(N, 1, 'last');
A = A(M,N,:);
if nargout > 1
    % Crop the map
    alpha = alpha(M,N);
end
return

function A = inpaint_background(A, alpha)
% Inpaint some of the background pixels with the colour of the nearest
% foreground neighbour
% Create neighbourhood
[Y X] = ndgrid(-4:4, -4:4);
X = Y .^ 2 + X .^ 2;
[X I] = sort(X(:));
X(I) = 2 .^ (numel(I):-1:1); % Use powers of 2
X = reshape(single(X), 9, 9);
X = X(end:-1:1,end:-1:1); % Flip for convolution
% Convolve with the mask & compute closest neighbour
M = conv2(single(alpha), X, 'same');
J = find(M ~= 0 & ~alpha);
[M M] = log2(M(J));
% Compute the index of the closest neighbour
[Y X] = ndgrid(-4:4, (-4:4)*size(alpha, 1));
X = X + Y;
X = X(I);
M = X(numel(X) + 2 - M) + J;
% Reshape for colour transfer
sz = size(A);
A = reshape(A, [sz(1)*sz(2) sz(3)]);
% Set background pixels to white (in case figure is greyscale)
A(~alpha,:) = 255;
% Change background colour to closest foreground colour
A(J,:) = A(M,:);
% Reshape back
A = reshape(A, sz);
return

function b = isvector(formats)
b = formats.pdf || formats.eps;
return

function b = isbitmap(formats)
b = formats.png || formats.tif || formats.jpg || formats.bmp;
return
