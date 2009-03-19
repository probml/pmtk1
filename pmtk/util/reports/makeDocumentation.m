function makeDocumentation(version,destination)
% Make all of the PMTK documentation, including published examples,
% contributing author report, and UML class diagram. The default
% destination is 'C:\kmurphy\pmtkLocal\docs\docxxx' where xxx is given by
% the specified version string, e.g. '1.4.2'. If specifying another
% destination, do not include the final docxxx directory. 
    
    if(nargin == 0)
       error('You must specify a version, e.g. makeDocumentation 1.4.2'); 
    end
    %% Setup Directory
    currentDir = pwd;
    if nargin < 2,destination = 'C:\kmurphy\pmtkLocal\docs\';  end
    destination = [destination,'doc',version];
    try cd(destination); 
        button = questdlg(sprintf('Documentation for version %s already exists.\n%sDo you want to regenerate it?',version,blanks(11)),'PMTK','Yes','Cancel','Cancel');
        if strcmpi(button,'Cancel'),return;end       
    catch                                                                  %#ok
        if(system(['mkdir ',destination])),error('Unable to create destination directory at %s',destination);end;
        cd(destination);  
    end
    %%
    fprintf('Publishing Examples...\n');
    publishExamples(destination);
    fprintf('Generating author report...\n');
    makeAuthorReport(fullfile(destination,'authors'));
    fprintf('Generating class diagram...\n');
    makeClassTree(destination);
    cd(destination); cd ..;
    fprintf('Zipping up contents...\n');
    zip(['doc',version],['.\doc',version,'\']);
    cd(currentDir);
    cls; clc;
    fprintf('Done generating PMTK documentation\n');
   
    
end