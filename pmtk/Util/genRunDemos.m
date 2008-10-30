function genRunDemos(className)
% This script automatically generates the runDemos.m file, which contains
% code to run selected BLT demos. It searches through every class method
% and includes those methods that meet the criteria specified in the
% subfunction include(). If includeScripts, (below) is true, it also
% searches through every non-class m-file and includes scripts with the
% include directive #demo.
%
% If className is specified, only demos from that class are listed and the
% runDemos file name is appended with the className.
% 
% Each included method is written to the runDemos file as in
%
%    methodName1(className1);
%    methodName2(className2);
% 
%    or, if static as
%
%   className1.methodName1();
%   className2.methodName2();
%
% and must therefore be capable of running as such. No checks are performed
% to ensure that this is the case. 
%
% If runDemos.m already exists, it is renamed runDemos.old. If there is
% already a runDemos.old file, it is overwritten. 
%
% This function will only work on windows systems. Run this script from the
% top level BLT directory.
%
% Version 5
    error('Deprecated - use makeRunDemos() instead');
    maxDemoCalls = 10;              % add the commands specified in addCommands after
                                    % every 10 demo calls. 
    function addCommands(fid)
    % Add these commands to the file at specified points
        %fprintf(fid,'\nplaceFigures;\n');
        fprintf(fid,'pause(2)\n');
        fprintf(fid,'close all\n');
        fprintf(fid,'clear all\n\n');
    end


filename = 'runDemos';      %The name of the generated m-file.
rootdir  = '.';             %Start searching from the current directory.
includeDirective = '#demo'; %Look for this directive in function comments.
includeScripts = true;      %If true, non-method scripts with #demo tag are also included.


function bool = include(method,classname)
%Include methods satisfying these criteria. method is a structure
%holding info about the underlying method; classname is a string.
        pub     =  strcmpi(method.Access,'public');              %Make sure its public
        notabs  = ~method.Abstract;                              %Make sure its implemented
        local   =  strcmp(method.DefiningClass.Name,classname);  %Don't grab superclass definitions
        demo    =  strncmpi(method.Name,'demo',4);               %method name begins with 'demo'                 
        static  =  method.Static;                                %method is static 
        force   =  strcmpi(includeDirective,strtok(help([classname,'.',method.Name])));
                                                                 %#demo directive in comment
        
        bool = (pub && notabs && local) && ( (demo && static) || force);             
end

counter = 0;
if(nargin > 0)
     filename = [filename,className];
end
renameold(filename);
fid = openfile(filename);
[classes,fileinfo] = getclasses(rootdir);
if(nargin > 0)
    classes = {className};
end
writedemos(fid,classes);
if(includeScripts)
    writescripts(fid,fileinfo);
end;
addCommands(fid);
closefile(fid);


   

function writedemos(fid,classes)
%Search through every class for methods satisfying the include statements
%and write calling syntax to the open file, (fid).
    for i=1:numel(classes)
        try
            meta = eval(['?',classes{i}]);
            methods = meta.Methods;
            for m =1: numel(methods)
                method = methods{m};
                if(include(method,classes{i}))
                    if(method.Static)
                        fprintf(fid,[classes{i},'.',method.Name,'();\n']);  
                    else
                        fprintf(fid,[method.Name,'(',classes{i},');\n']);
                    end
                    counter = counter + 1;
                    if (counter >= maxDemoCalls)
                        addCommands(fid);
                        counter = 0;
                    end
                end
            end
        catch ME
            warning('CLASSTREE:discoveryWarning',['Could not discover information about class ',classes{i}]);
            continue; %Keep going, even if there's an error. 
        end
    end
end

function writescripts(fid,fileinfo)
%Search through every m-file looking for the include directive. If found,
%write calling syntx to the open file, (fid).
    mfiles = vertcat(fileinfo.m);
    for m=1:numel(mfiles)
        [path,name,ext,versn] = fileparts(mfiles{m});
        if(strcmpi(includeDirective,strtok(help(name))))
            fprintf(fid,[name,';\n']);
            counter = counter + 1;
            if (counter >= maxDemoCalls)
                addCommands(fid);
                counter = 0;
            end
        end
    end

end

    
function [classes,info] = getclasses(directory)
%Return the names of all of the classes in the specified directory and all
%of its subdirectories. 
    info = dirinfo(directory);
    classes = vertcat(info.classes); 
end

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

function fid = openfile(filename)
    fid = fopen([filename,'.m'],'w');
    fprintf(fid,'%%Code automatically generated by genRunDemos.\n%%Run BLT demos.\n');
end

function closefile(fid)
    fprintf(fid,'\n');
    fclose(fid);
end

function renameold(filename)
%Rename existing output file, if any to filename.old
    flist = dir;
    files = {flist.name};
    if(ismember([filename,'.m'],files))
        fprintf(['\nrenaming ',filename,'.m as ',filename,'.old ...\n']);
        if(ismember([filename,'.old'],files))
            eval(['!del ',filename,'.old']);
        end
        eval(['!rename ',filename,'.m ',filename,'.old']);
    end
end

end