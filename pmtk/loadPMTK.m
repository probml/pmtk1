%% Initialize the PMTK path
% Note, this only initializes the path for the current Matlab session.

% You can add PMTK to your path permanently by typing "pathtool" at the prompt, 
% selecting "add with subfolders" and choosing the root PMTK directory. Note, 
% however, that the directory structure may change from version to version. Before
% upgrading, you should remove the PMTK path entries via pathtool and then 
% add them again once the new version is in place. If you have added PMTK via
% pathtool, removing it via unloadPMTK will not suffice. 


cd(fileparts(which('loadPMTK.m')));   % Make sure the current directory is the root of PMTK
addpath(fullfile(pwd,'util'));        % Add util first to get access to genpathPMTK.m
addpath(genpathPMTK(pwd));            % Add all subdirectories, (except for svn, old, etc)
fprintf('Welcome to PMTK\n');