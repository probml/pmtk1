function makeClassTree(location)
% Use viewClassTree to generate a class UML diagram for PMTK and save it
% to a file. 
    filename = 'classDiagram';
    if(nargin == 0)
        location = 'C:\kmurphy\pmtkLocal\doc\';
    end
    
    cd(PMTKroot());
    h = viewClassTree();
    maximizeFigure();
    pause(1);
    tightenAxes(h);
    shrinkNodes(h);
    for i=1:3
        increaseFontSize(h);
    end
    %print('-dpng',fullfile(location,filename));
    print_pdf(fullfile(location,filename));
    close all;
end