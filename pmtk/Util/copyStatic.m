function copyStatic(source,destination)  
% A combination of the genRunDemos function and the publishMethod function. It 
% copies all of the static methods whose names begin with "demo" to their own
% stand alone scripts under the specified destination. 

error('Incomplete');


[classes,fileinfo] = getclasses(source);

return;



function bool = include(method,classname)
%Include methods satisfying these criteria. method is a structure
%holding info about the underlying method; classname is a string.
        pub     =  strcmpi(method.Access,'public');              %Make sure its public
        notabs  = ~method.Abstract;                              %Make sure its implemented
        local   =  strcmp(method.DefiningClass.Name,classname);  %Don't grab superclass definitions
        demo    =  strncmpi(method.Name,'demo',4);               %method name begins with 'demo'                 
        static  =  method.Static;                                %method is static 
        %force   =  strcmpi(includeDirective,strtok(help([classname,'.',method.Name])));
                                                                 %#demo directive in comment
        
        bool = pub && notabs && local && demo && static;             
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
    
    
    
end