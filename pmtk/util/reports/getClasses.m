function classes = getClasses(source)
% Get a list of all of the classes below the specified directory that are 
% on the Matlab path.
    
if nargin < 1
    source = '.';
end
excludeList = {'dependsOn','viewClassTree','getClasses'};
classes = setdiff(findClasses(dirinfo(source)),excludeList)';


function baseClasses = findClasses(info)
   baseClasses = {}; 
   for i=1:numel(info)
      mfiles = info(i).m;
      for j=1:numel(mfiles)
          file = mfiles{j};
          if ~exist(file(1:end-2),'file'), continue; end
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