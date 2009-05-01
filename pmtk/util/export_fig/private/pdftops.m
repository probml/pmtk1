function varargout = pdftops(cmd)
%PDFTOPS  Calls a local pdftops executable with the input command
%
% Example:
%   [status result] = pdftops(cmd)
%
% Attempts to locate a pdftops executable, finally asking the user to
% specify the directory pdftops was installed into. The resulting path is
% stored for future reference.
% 
% Once found, the executable is called with the input command string.
%
% This function requires that you have pdftops (from the Xpdf package)
% installed on your system. You can download this from:
% Xpdf: http://www.foolabs.com/xpdf
%
% IN:
%   cmd - Command string to be passed into pdftops.
%
% OUT:
%   status - 0 iff command ran without problem.
%   result - Output from pdftops.

% $Id: pdftops.m,v 1.4 2009/04/09 11:11:23 ojw Exp $
% Copyright: Oliver Woodford, 2009

% Call pdftops
str = sprintf('"%s" %s', xpdf_path, cmd);
if nargout
    [varargout{:}] = system(str);
else
    system(str);
end
return

function path = xpdf_path
% Return a valid path
% Start with the currently set path
path = current_xpdf_path;
% Check the path works
if check_xpdf_path(path)
    return
end
% Check whether the binary is on the path
if ispc
    %bin = 'pdftops.exe';
    bin = 'pdftops';
else
    bin = 'pdftops';
end
if check_store_xpdf_path(bin)
    path = bin;
    return
end
% Search the obvious places
if ispc
    %path = 'C:\Program Files\xpdf\pdftops.exe';
    path = '/usr/local/bin/pdftops';
else
    path = '/usr/local/bin/pdftops';
end
if check_store_xpdf_path(path)
    return
end
% Ask the user to enter the path
while 1
    base = uigetdir('/', 'Pdftops not found. Please select pdftops installation directory.');
    if isequal(base, 0)
        % User hit cancel or closed window
        break;
    end
    base = [base filesep];
    bin_dir = {'', ['bin' filesep], ['lib' filesep]};
    for a = 1:numel(bin_dir)
        path = [base bin_dir{a} bin];
        if exist(path, 'file') == 2
            break;
        end
    end
    if check_store_xpdf_path(path)
        return
    end
end
error('pdftops executable not found.');

function good = check_store_xpdf_path(path)
% Check the path is valid
good = check_xpdf_path(path);
if ~good
    return
end
% Update the current default path to the path found
fname = which(mfilename);
% Read in the file
fh = fopen(fname, 'rt');
fstrm = fread(fh);
fclose(fh);
% Find the path
fstrm = char(fstrm');
first_sec = regexp(fstrm, '[\n\r]function path = current_xpdf_path[\n\r]path = ''', 'end', 'once');
second_sec = first_sec + regexp(fstrm(first_sec+1:end), ''';[\n\r]return', 'once');
% Save the file with the path replaced
fh = fopen(fname, 'wt');
fprintf(fh, '%s%s%s', fstrm(1:first_sec), path, fstrm(second_sec:end));
fclose(fh);
return

function good = check_xpdf_path(path)
% Check the path is valid
[good message] = system(sprintf('"%s" -h', path));
good = good == 1;
return

function path = current_xpdf_path
path = 'pdftops.exe';
path = 'pdftops';
return
