function [classes,dirInfo] = getClasses(source)
% Get a list of all of the classes below the specified directory. These must
% use the classdef syntax and reside in their own @ directories
% Info returns info on the entire directory structure and its files.     
    
dirInfo = dirinfo(source);
classes = vertcat(dirInfo.classes); 


function info = dirinfo(directory)
%Get info about all of the files in the directory structure. 
    info = what(directory);
    flist = dir(directory);
    dlist =  {flist([flist.isdir]).name};
    for i=1:numel(dlist)
        dirname = dlist{i};
        if(~strcmp(dirname,'.') && ~strcmp(dirname,'..'))
            info = [info, dirinfo([directory,'\',dirname])]; 
        end
    end
end
    
    
    
end