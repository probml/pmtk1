function makeDocumentation(destination)
    % This function automatically generates documentation for the PMTK system.

    demodirectory = './examples';

    originalDirectory = pwd;                    % save current directory
    if(nargin == 0), destination = 'C:\PMTKdocs';  end

    cd(PMTKroot());
    demoDirInfo = dirinfo(demodirectory);
    destRoot = makeDestinationDirectory();
    rootHTMLfid = prepareRootHTMLfile();


    for i=1:numel(demoDirInfo)
        entry = demoDirInfo(i);
        if(~isempty(entry.m))
            [base,link] = fileparts(entry.path);
            linkName = link;
            if(~isempty(strfind(link,'Examples')))
                linkName = linkName(1:end-8);
            end
            fprintf(rootHTMLfid,'%%%%\n%% <html>\n%% <hr>\n%% </html>\n%%%%\n');
            fprintf(rootHTMLfid,'\n%%%% %s\n',[linkName(1:end) ,' Demos']);
            for j=1:numel(entry.m)
                fileToPublish = entry.m{j};
                fid = fopen(which(fileToPublish));
                fulltext = textscan(fid,'%s','delimiter','\n','whitespace','');
                fulltext = fulltext{:};
                txt = fulltext{1};
                [start,rest] = strtok(txt,' ');
                if(strcmp(start,'%%'))
                    txt = rest;
                else
                    txt = fileToPublish(1:end-2);
                end
                fclose(fid);
                
                writeHTMLlink(rootHTMLfid,['./',link,'/',fileToPublish(1:end-2),'.html'],txt);
                
              
                fileToPublish = fileToPublish(1:end-2);
                outputDir = fullfile(destination,destRoot,link);
                cd(entry.path);
                publishFile(fileToPublish,outputDir,true);
                close all;
            end
            
        end
    end
    fprintf(rootHTMLfid,'%%%%\n%% <html>\n%% <hr>\n%% </html>\n%%%%\n');
    fclose(rootHTMLfid);
    cd(fullfile(destination,destRoot));
    publishFile('PMTKdocs.m','.',false);
    delete('PMTKdocs.m');


    cd(originalDirectory);                     % restore current directory
    cls;


    function publishFile(mfile,outputDir,evalCode)
        options.evalCode = evalCode;
        options.outputDir = outputDir;
        options.format = 'html';
        options.createThumbnail = false;
        publish(mfile,options);

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


    function fid = prepareRootHTMLfile()
        fid = fopen('PMTKdocs.m','w+');
        fprintf(fid,'%%%% PMTK Documentation\n')
        d = date;
        fprintf(fid,'%% Revision Date: %s\n\n',d);

    end

    function writeHTMLlink(fid,link,name)
        fprintf(fid,'%%%%\n');
        fprintf(fid,'%% <html>\n');
        fprintf(fid,'%% <A HREF="%s">%s</A><br>\n',link,name);
        fprintf(fid,'%% </html>\n');
        fprintf(fid,'%%%%\n');
    end

  


end