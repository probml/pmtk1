function makeRunDemos()
% This function automatically generates the runDemos.m file, which contains
% code to run all of the PMTK scripts located in the examples directory. 
% Various tags can be used to exclude certain scripts.
    
    excludeTag = '%#exclude';       % do not include in runDemos
    slowTag    = '%#slow';          % include  but comment with slow tag
    brokenTag  = '%#broken';        % include  but comment with broken tag
    inprogressTag = '%#inprogress'; % include but comment with in progress tag

    reportDir = fileparts(which('makeRunDemos'));
    cd(reportDir);  cd ..;
    
    if(exist('runDemos.m','file'))
        if(exist('runDemos.old','file'))
           delete runDemos.old 
        end
        !rename runDemos.m runDemos.old
    end
    fid = fopen('runDemos.m','w+');
    fprintf(fid,'%%%% Run Every Demo\n\n');
    
    info = dirinfo('../examples');
    
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
                        inprogress = tagsearch(entry.m{j},inprogressTag);
                        if(slow || broken || inprogress), fprintf(fid,'%%'); end
                        mfile = entry.m{j};
                        fprintf(fid,'%s',mfile(1:end-2));
                        if(broken)
                            fprintf(fid,' - broken!\n');
                        elseif(slow)
                            fprintf(fid,' - slow!\n');
                        elseif(inprogress)
                            fprintf(fid,' - not yet finished\n');
                        else
                            fprintf(fid,';%spclear;\n',blanks(max(5,40-length(mfile))));
                        end        
                    end
                    
             end
             
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
