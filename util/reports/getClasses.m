function classes = getClasses(source)
% Get a list of all of the classes below the specified directory.     
    
if nargin < 1
    source = '.';
end
excludeList = {'dependsOn','viewClassTree','getClasses'};
classes = setdiff(findClasses(dirinfo(source)),excludeList)';

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


function baseClasses = findClasses(info)
   baseClasses = {}; 
   for i=1:numel(info)
      mfiles = info(i).m;
      for j=1:numel(mfiles)
          file = mfiles{j};
          fid = fopen(file);
          fulltext = textscan(fid,'%s','delimiter','\n','whitespace','');
          fclose(fid);
          fulltext = fulltext{:};
          if(~isempty(cell2mat(strfind(fulltext,'classdef'))))
              baseClasses = [baseClasses;file(1:end-2)];
          end
      end
   end
end
    
    
    
end