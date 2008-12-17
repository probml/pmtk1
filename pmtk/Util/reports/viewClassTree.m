function  h=viewClassTree(directory)
% View a class inheritence hierarchy. All classes residing in the directory
% or any subdirectory are discovered. Parents of these classes are also
% discovered as long as they are in the matlab search path. 
% There are a few restrictions:
% (1) classes must be written using the new 2008a classdef syntax
% (2) requires the Graphlayout package be on the path to display the tree.
% (3) may not work on non-windows systems.
%  
% directory  is an optional parameter specifying the base directory of the
%            project. The current working directory is used if this is not specified.
%Written by Matthew Dunham 
%
% Changes:
%
% November 4,2008 - modified to work when classes do not reside in @
% directories. 


if nargin == 0
    directory = '.';
end
excludeList = {'dependsOn','viewClassTree','getClasses'};

info = removeUnwanted(dirinfo(directory));
errors = {};
baseClasses = findClasses(info);

if(isempty(baseClasses))
    fprintf('\nNo classes found in this directory.\n');
    return;
end

allClasses = baseClasses;
for c=1:numel(baseClasses)
   allClasses = union(allClasses,ancestors(baseClasses{c}));
end

matrix = zeros(numel(allClasses));
map = struct;
for i=1:numel(allClasses)
   map.(allClasses{i}) = i; 
end

markForDeletion = [];
for i=1:numel(allClasses)
    if(isempty(cell2mat(strfind(excludeList,allClasses{i}))))
        try
            meta = eval(['?',allClasses{i}]);
            parents = meta.SuperClasses;
        catch ME
            errors = [errors;allClasses{i}];
            markForDeletion = [markForDeletion,i];
            continue;
        end
        for j=1:numel(parents)
            matrix(map.(allClasses{i}),map.(parents{j}.Name)) = 1;
        end
    else
        markForDeletion = [markForDeletion,i];
    end
end
allClasses(markForDeletion) = [];
matrix(markForDeletion,:) = [];
matrix(:,markForDeletion) = [];

shortClassNames = shortenClassNames(allClasses);

h = Graphlayout('adjMatrix',matrix,'nodeLabels',shortClassNames,'splitLabels',true);

if(~isempty(errors))
    fprintf('\nThe following m-files were\nthought to be classes\nbecause they contain the\nclassdef keyword, but did\nnot respond to queries.\nThey were not included in the graph.\n\n');
    for i=1:numel(errors) 
       fprintf('%s\n',errors{i}); 
    end
end

% biog = biograph(matrix,allClasses);
% h=view(biog);
% set(h,'layouttype', 'equilibrium')
% dolayout(h)



function info = dirinfo(directory)
%Recursively generate an array of structures holding information about each
%directory/subdirectory beginning, (and including) the initially specified
%parent directory. 
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

function list = ancestors(class)
%Recursively generate a list of all of the superclasses, (and superclasses
%of superclasses, etc) of the specified class. 
    list = {};
    try
        meta = eval(['?',class]);
        parents = meta.SuperClasses;
    catch
        return;
    end
    for i=1:numel(parents)
       list = [list,parents{i}.Name]; 
    end
    
    for p=1:numel(parents)
        if(p > numel(parents)),continue,end %bug fix for version 7.5.0 (2007b)
        list = [list,ancestors(parents{p}.Name)];
    end
end

function info = removeUnwanted(info)
    unwanted = {'old','Old','OLD','deprecated','.svn'};
    for i=1:numel(unwanted)
        info(cell2mat(cellfun(@(str)~isempty(str),strfind({info.path} ,unwanted{i}),'UniformOutput',false))) = [];
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


function classNames = shortenClassNames(classNames)
    remove = {'Dist'};            % add to this list to remove other partial strings - case sensitive
    for i=1:numel(remove)
        ndx = strfind(classNames,remove{i});
        for j=1:numel(classNames)
           if(~isempty(ndx{j}))
               classNames{j}(ndx{j}:ndx{j}+length(remove{i})-1) = [];
           end
        end
    end
    
end

end