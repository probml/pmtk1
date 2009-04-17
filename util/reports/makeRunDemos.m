function makeRunDemos()
% This function automatically generates the runDemos.m file, which contains
% code to run all of the PMTK scripts located in the examples directory. 
% All of the work is done in processExamples(). 

    fname = 'runDemos.m';                    % name of the file to create
    includeTags = {};                        % include everything
    excludeTags = {'#broken','#inprogress'}; % commented out in runDemos.m
    pauseTime = 2;                           % time in seconds to pause between consecutive demos

    text = processExamples(includeTags,excludeTags,pauseTime);
    header = {'%% Run Every Demo'};
    footer = {''};
    text = [header;text;footer];
    writeText(text,fullfile(PMTKroot(),'util',fname)); 


end