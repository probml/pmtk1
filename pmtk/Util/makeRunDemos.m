function makeRunDemos()
% This function automatically generates the runDemos.m file, which contains
% code to run all of the PMTK demos located in the examples directory. 
%
% Version 1.0
    
    utilDir = fileparts(which('makeRunDemos'));
    cd(utilDir);  cd ..;
    
    if(exist('runDemos.m','file'))
        if(exist('runDemos.old','file'))
           delete runDemos.old 
        end
        !rename runDemos.m runDemos.old
    end
    fid = fopen('runDemos.m','w+');
    fprintf(fid,'%%%% Run Every Demo\n\n');
    
    info = dirinfo('./examples');
    
    for i=1:numel(info)
       entry = info(i);
       if(~isempty(entry.m))
             [base,pack] = fileparts(entry.path);
             tl = '';
             if(length(path) > 8 && strfind(pack,'Examples'))
                tl = pack(1:end-8);  
             end
           
             fprintf(fid,'%%%% %s\n',tl);
             for j=1:numel(entry.m)
                    mfile = entry.m{j};
                    fprintf(fid,'%s\n',mfile(1:end-2));
             end
             fprintf(fid,'pause(2); close(''all''); clear(''all'');\n\n');           
       end
        
    end
   
    fclose(fid);
    
    
    
    
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
