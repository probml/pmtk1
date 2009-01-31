function makeRunDemos()
% This function automatically generates the runDemos.m file, which contains
% code to run all of the PMTK demos located in the examples directory. 
%
% Version 1.0
    
    excludeTag = '%#exclude';   % do not include in runDemos
    slowTag    = '%#slow';      % include in runDemos but comment out with slow tag
    brokenTag  = '%#broken';    % include in runDemos but comment out with broken tag

    reportDir = fileparts(which('makeRunDemos'));
    cd(reportDir);  cd ..;cd ..;
    
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
             if(length(pack) > 8 && strfind(pack,'Examples'))
                tl = pack(1:end-8);  
             end
           
             fprintf(fid,'%%%% %s\n',tl);
             for j=1:numel(entry.m)
                    if(~tagsearch(entry.m{j},excludeTag))
                        slow = tagsearch(entry.m{j},slowTag);
                        broken = tagsearch(entry.m{j},brokenTag);
                        if(slow || broken), fprintf(fid,'%%'); end
                        mfile = entry.m{j};
                        fprintf(fid,'%s',mfile(1:end-2));
                        if(broken)
                            fprintf(fid,' - broken!\n');
                        elseif(slow)
                            fprintf(fid,' - slow!\n');
                        else
                            fprintf(fid,'\n');
                        end 
                    end
                    
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
