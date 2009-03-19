function makeTestPMTK()
% This function automatically generates the testPMTK.m file, which contains
% code to run all of the PMTK tests located in the examples directory. A
% test is a script with the #testPMTK tag. All of the work in generating
% this file is done within processExamples().

    fname        = 'testPMTK.m';                      % name of the file to create
    includeTags = {'#testPMTK'};                      % include only tests
    excludeTags = {'#broken','#inprogress','#slow'};  % commented out in runDemos.m
    pauseTime = 0;                                    % time in seconds to pause between consecutive demos

    text = processExamples(includeTags,excludeTags,pauseTime);
    
    header = {'%% Test PMTK';'';'try';''};            
    footer = {'objectCreationTest; % try instantiating every class...'
              'pclear(0);' 
              ''
              'catch ME'
              'clc; close all'
              'fprintf(''PMTK Tests FAILED!\npress enter to see the error...\n\n'');'
              'pause'
              'rethrow(ME)'
              'end'
              ''
              'cls'
              'clc'
              'fprintf(''PMTK Tests Passed\n'')'
              };
              
    text = [header;text;footer];
    writeText(text,fullfile(PMTKroot(),'util',fname)); 


end
   