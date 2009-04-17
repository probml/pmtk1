%% Remove PMTK from the Matlab path
%  Exactly undoes what loadPMTK() does. This script does not delete any files. 

if(exist('genpathPMTK','file'))
s = warning('query','all');
warning off all
rmpath(genpathPMTK(PMTKroot()))
warning(s);
fprintf('PMTK removed from the Matlab path\n');
else
    cd(fileparts(which('unloadPMTK.m')));   % Make sure the current directory is the root of PMTK
    addpath(fullfile(pwd,'util'));
    if(~exist('genpathPMTK','file'))
       error('Could not remove PMTK from the path - missing genpathPMTK()'); 
    end
    unloadPMTK;
end