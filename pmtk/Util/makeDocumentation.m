function errors = makeDocumentation(destination)
    % This function automatically generates documentation for the PMTK system.
    
    demodirectory = './examples';               
    
    
    originalDirectory = pwd;                    % save current directory
    if(nargin == 0), destination = 'C:\PMTKdocs';  end

    errors = {};
    cd(PMTKroot());
    demoDirInfo = dirinfo(demodirectory);
    destRoot = makeDestinationDirectory();
    
    for i=1:numel(demoDirInfo)
        entry = demoDirInfo(i);
        if(~isempty(strfind(entry.path,'+')) && isempty(strfind(entry.path,'.svn'))&& ~isempty(entry.m))
            for j=1:numel(entry.m)
                fileToPublish = fullfile(entry.path,entry.m{j});
                [base,pack] = fileparts(entry.path);
                outputDir = fullfile(destRoot,pack);
                publishFile(fileToPublish,outputDir,true);
            end
        end
    end

    cd(originalDirectory);                     % restore current directory


    function publishFile(mfile,outputDir,evalCode)
          options.evalCode = evalCode;
          options.outputDir = outputDir;
          options.format = 'html';
          try
            publish([methodName,'Published.m'],options);
          catch ME
            display(ME.message);
            errors = {errors;mfile};
          end
    end




    function destRoot =  makeDestinationDirectory()
        % Create the root directory for the documentation.
        try(cd(destination))   % See if it already exists
        catch                  % if not, create it
            err = system(['mkdir ',destination]);
            if(err)            % if could not create it, error
                error('Unable to create destination directory at %s',destination);
            end
            cd(destination);   % go to the new directory
        end
        d = date;              % current date
        dirinfo = dir;         % see if this subdirectory already exists
        appendTime = ~isempty(cell2mat(strfind({dirinfo.name},d))); % if it does, append the time
        destRoot = d;
        if(appendTime)
            destRoot = [destRoot,'-',num2str(rem(now,1)*24)];
        end
        err = system(['mkdir ',destRoot]);   % create the subdirectory
        if(err)
            error('Unable to create destination directory at %s\%s',destination,destRoot);
        end
        cd(destRoot);
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


end