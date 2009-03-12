function makeTestPMTK()
% This function automatically generates the testPMTK.m file, which contains
% code to run all of the PMTK tests located in the examples directory. A test
% is a demo with the %#testPMTK tag. 
%
% Version 1.0
    
    includeTag    = '%#testPMTK';
    slowTag       = '%#slow';       % include in testPMTK but comment out with slow tag
    brokenTag     = '%#broken';     % include in testPMTK but comment out with broken tag
    inprogressTag = '%#inprogress'; % include in testPMTK but commment out with not yet finished tag

    reportDir = fileparts(which('makeTestPMTK'));
    cd(reportDir);  cd ..;    % store in util directory, one up from reports
    
   
    fid = fopen('testPMTK.m','w+');
    fprintf(fid,'%%%% Test PMTK\n\n');
    fprintf(fid,'try\n');
    
    info = dirinfo('../examples');
    
    for i=1:numel(info)
       entry = info(i);
       tests = entry.m;
       include = false(1,numel(tests));
       for t=1:numel(tests)
          include(t) = tagsearch(tests{t},includeTag);
       end
       tests = tests(include);
       if(~isempty(tests))
             [base,pack] = fileparts(entry.path);
             tl = '';
             if(length(pack) > 8 && strfind(pack,'Examples'))
                tl = pack(1:end-8);  
             end
             fprintf(fid,'%%%% %s\n',tl);
             for j=1:numel(tests)
                 testFile = tests{j};
                 slow = tagsearch(testFile,slowTag);
                 broken = tagsearch(testFile,brokenTag);
                 inprogress = tagsearch(testFile,inprogressTag);
                 if(slow || broken || inprogress), fprintf(fid,'%%'); end
                 fprintf(fid,'%s',testFile(1:end-2));
                 if(broken)
                     fprintf(fid,' - broken!\n');
                 elseif(slow)
                     fprintf(fid,' - slow!\n');
                 elseif(inprogress)
                     fprintf(fid,' - not yet finished\n');
                 else
                     fprintf(fid,';%spclear(0);\n',blanks(max(5,40-length(testFile))));
                 end
             end
             
       end
    end
    fprintf(fid,'%% try instantiating every class...\n');
    fprintf(fid,'objectCreationTest\n\n');
    fprintf(fid,'pclear(0);\n\n');
    fprintf(fid,'catch ME\n');
    fprintf(fid,'\tclc; close all\n');
    fprintf(fid,'\tfprintf(''PMTK Tests FAILED!\\npress enter to see the error...\\n\\n'')\n');
    fprintf(fid,'\tpause\n');
    fprintf(fid,'\trethrow(ME)\n');
    fprintf(fid,'end\n\n');
    fprintf(fid,'cls\n');
    fprintf(fid,'fprintf(''PMTK Tests Passed\\n'')');
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
