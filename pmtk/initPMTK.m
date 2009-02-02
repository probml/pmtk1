%% Initialize the PMTK path

cd(fileparts(which('initPMTK.m')));   % Make sure the current directory is the root of PMTK
addpath(fullfile(pwd,'util'));        % Add util first to get access to genpathPMTK.m
addpath(genpathPMTK(pwd));            % Add all subdirectories, (except for svn, old, etc)
fprintf('Welcome to PMTK\n');