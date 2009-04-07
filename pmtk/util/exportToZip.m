function zipfilename = exportToZip(funcname,zipfilename)
%
%EXPORTTOZIP - Creates a ZIP file containing all dependencies of a function
% zipfilename = exportToZip(funcname)
% zipfilename = exportToZip(funcname,zipfilename)
%
% All M-files which are required by the specified function(s) and which are
% not inside MATLAB's "toolbox" directory.
% funcname is one or more function names, as a string or cell array of strings.
%
% If no zipfilename is specified, the ZIP file is created in the current
% working directory with name X.zip where X is the first function name in
% the supplied list.  The file names in the ZIP file are relative to the
% common root directory directory of all the required files.  If no
% common root directory can be established (e.g. if files are on
% different drives) an error is thrown.
%
%#author Malcolm Wood
%#url http://www.mathworks.com/matlabcentral/fileexchange/10702
%
% Modified by Matt Dunham to also exclude PMTK
%
%
if ~iscell(funcname)
    funcname = {funcname};
end

if isempty(funcname)
    error('No function names specified');
end

if ~iscellstr(funcname)
    error('Function names must be strings');
end

if nargin<2
    zipfilename = fullfile(pwd,[ funcname{1} '.zip' ]);
end

req = cell(size(funcname));
for i=1:numel(funcname)
    req{i} = mydepfun(funcname{i},1); % recursive
end
req = vertcat(req{:}); % cell arrays of full file names
req = unique(req);

% Find the common root directory
d = i_root_directory(req);
% Calculate relative paths for all required files.
n = numel(d);
for i=1:numel(req)
    % This is the bit that can't be vectorised
    req{i} = req{i}(n+1:end); % step over last character (which is the separator)
end

zip(zipfilename,req,d);
fprintf(1,'Created %s with %d entries\n',zipfilename,numel(req));

%%%%%%%%%%%%%%%%%%%%%
% Identifies the common root directory of all files in cell array "req"
function d = i_root_directory(req)

d = i_parent(req{1});
for i=1:numel(req)
    t = i_parent(req{i});
    if strncmp(t,d,numel(d))
        % req{i} is in directory d.  Next file.
        continue;
    end
    % req{i} is not in directory d.  Up one directory.
    count = 1;
    while true
        % Remove trailing separator before calling fileparts.  Add it
        % again afterwards.
        tempd = i_parent(d(1:end-1));
        if strcmp(d,tempd)
            % fileparts didn't find us a higher directory
            error('Failed to find common root directory for %s and %s',req{1},req{i});
        end
        d = tempd;
        if strncmp(t,d,numel(d))
            % req{i} is in directory d.  Next file.
            break;
        end
        % Safety measure for untested platform.
        count = count+1;
        if count>1000
            error('Bug in i_root_directory.');
        end
    end
end

%%%%%%%%%%%%%%%%%%%
function d = i_parent(d)
% Identifies the parent directory, including a trailing separator

% Include trailing separator in all comparisons so we don't assume that
% file C:\tempX\file.txt is in directory C:\temp
d = fileparts(d);
if d(end)~=filesep
    d = [d filesep];
end

function filelist = mydepfun(fn,recursive)
%MYDEPFUN - Variation on depfun which skips toolbox files
%
% filelist = mydepfun(fn)
% filelist = mydepfun(fn,recursive)
%
% Returns a list of files which are required by the specified
% function, omitting any which are inside $matlabroot/toolbox. 
%
% "fn" is a string specifying a filename in any form that can be
%   identified by the built-in function "which".
% "recursive" is a logical scalar; if false, only the files called
%   directly by the specified function are returned.  If true, *all*
%   those files are scanned to, and any required by those, and so on.
%
% "filelist" is a cell array of fully qualified file name strings,
%   including the specified file.
%
% e.g.
%     filelist = mydepfun('myfunction')
%     filelist = mydepfun('C:\files\myfunction.m',true) 

if ~ischar(fn)
    error('First argument must be a string');
end

foundfile = which(fn);
if isempty(foundfile)
    error('File not found: %s',fn);
end

% Scan this file
filelist = i_scan(foundfile);

% If "recursive" is supplied and true, scan files on which this one depends.
if nargin>1 && recursive
    % Create a list of files which we have still to scan.
    toscan = filelist;
    toscan = toscan(2:end); % first entry is always the same file again
    % Now scan files until we have none left to scan
    while numel(toscan)>0
        % Scan the first file on the list
        newlist = i_scan(toscan{1});
        newlist = newlist(2:end); % first entry is always the same file again
        toscan(1) = []; % remove the file we've just scanned
        % Find out which files are not already on the list.  Take advantage of
        % the fact that "which" and "depfun" return the correct capitalisation
        % of file names, even on Windows, making it safe to use "ismember"
        % (which is case-sensitive).
        reallynew = ~ismember(newlist,filelist);
        newlist = newlist(reallynew);
        % If they're not already in the file list, we'll need to scan them too.
        % (Conversely, if they ARE in the file list, we've either scanned them
        %  already, or they're currently on the toscan list)
        toscan = unique( [ toscan ; newlist ] );
        filelist = unique( [ filelist ; newlist ] );
    end
end

%%%%%%%%%%%%%%%%%%%%%
% Returns the non-toolbox files which the specified one calls.
% The specified file is always first in the returned list.
function list = i_scan(f)

func = i_function_name(f);

list = depfun(func,'-toponly','-quiet');
ulist = lower(list);

toolboxroot = lower(fullfile(matlabroot,'toolbox'));

intoolbox = strncmp(ulist,toolboxroot,numel(toolboxroot));



if exist('PMTKroot','file')
    inpmtk = strncmpi(ulist,PMTKroot(),numel(PMTKroot()));
    include = ~intoolbox & ~inpmtk;
    include(1) = true;  % don't exclude root function!
    list = list(include);
else
    list = list(~intoolbox);
end



%%%%%%%%%%%%%%%%%%%%%%%%
function func = i_function_name(f)
% Identifies the function name for the specified file,
% including the class name where appropriate.  Does not
% work for UDD classes, e.g. @rtw/@rtw

[dirname,funcname,ext] = fileparts(f);
[ignore,dirname] = fileparts(dirname);

if ~isempty(dirname) && dirname(1)=='@'
    func = [ dirname '/' funcname ];
else
    func = funcname;
end










