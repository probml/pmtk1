function makeDocumentation(destination)
% This function automatically generates documentation for the PMTK system.
% By default the root doc directory is set to C:\PMTKdocs but you can specify
% another as a parameter to this function. Within this root directory, it
% creates a directory structure and html files corresponding to the examples in
% the example directory.

    demodirectory = './examples';                               % location relative to the root of PMTK where the demos live
    viewName1     = 'PMTKdocsByFuncName.html';                  % name of root HTML file for the docs
    defaultDocDir = 'C:\PMTKdocs';                              % default directory to store the documentation
    doNotEvalTag  = '%#!';                                      % If this tag is present in the function's documentation, it is not evaluated when published
    excludeList = {};                                           % Functions listed here are not displayed in the "Functions Used" column
    if(exist('trivialFunctionList.txt','file'))
        excludeList = [excludeList,getText('trivialFunctionList.txt')];
    end

    makeRootOnly = false;                                       % If true, only the root html file is generated, nothing else is published. 
    
    originalDirectory = pwd;                                    % save current directory
    if(nargin == 0), destination = defaultDocDir;  end          % this is where the docs will live
    cd(PMTKroot());                                             % change to the base PMTK directory
    demoDirInfo = dirinfo(demodirectory);                       % collect information about the demos
    destRoot = makeDestinationDirectory();                      % make the documentation directory

    viewInfo = struct('primaryClass',{},'functionName',{},'title'        ,{} ,'description',{},...
                      'htmlLink'    ,{},'classesUsed' ,{},'functionsUsed',{} ,'evalCode'   ,{},...
                      'outputDir'   ,{}); % store table info here
    demoCounter = 1;                     
    for i=1:numel(demoDirInfo)                                  % for every directory
        entry = demoDirInfo(i);                                 
        if(~isempty(entry.m))                                   % if it contains m-files loop over them
            [exdirName,viewInfo(demoCounter).primaryClassName] = getPrimaryClass(entry.path);
            for j=1:numel(entry.m)                              % for every mfile
                 [viewInfo(demoCounter).functionName    ,...
                 viewInfo(demoCounter).title            ,...
                 viewInfo(demoCounter).description      ,...
                 viewInfo(demoCounter).functionsUsed    ,...
                 viewInfo(demoCounter).classesUsed      ,...
                 viewInfo(demoCounter).evalCode] = getDemoInfo(entry.m{j});
                
                 
             
                 viewInfo(demoCounter).htmlLink = ['./',exdirName,'/',viewInfo(demoCounter).functionName,'.html'];
                 viewInfo(demoCounter).outputDir = fullfile(destination,destRoot,exdirName);  % publish to here
                 demoCounter = demoCounter + 1;
                  
            end
       
        end
    end
    if(~makeRootOnly)
        pause(1);
        for i=1:numel(viewInfo)
            evalin('base','clear all'); 
            publishFile(viewInfo(i).functionName,viewInfo(i).outputDir,viewInfo(i).evalCode); 
            close all;
        end
    end
    
    cdDocBase();        
    createViews(viewInfo);                          % create root html files with main table, etc
    fclose all;
    cd(originalDirectory);                          % restore current directory
    close all                
    evalin('base','clear all');                     % clear the base workspace
    %% Subfunctions

    function [exdirName, className] = getPrimaryClass(path)
    % From a file path, extract the example directory name and the associated
    % class name.
        [base,exdirName] = fileparts(path);             %#ok
        className = exdirName;
        if(~isempty(strfind(className,'Examples')))
            className = className(1:end-8);
        end
    end

    function [fname,title,description,funcsUsed,classesUsed,evalCode] = getDemoInfo(mfile)
    % Get information about the specified demo.
        fname = mfile(1:end-2);
        fid = fopen(which(mfile));
        fulltext = textscan(fid,'%s','delimiter','\n','whitespace','');
        fulltext = fulltext{:};
        title = fulltext{1};
        [start,rest] = strtok(title,' ');
        title = '';
        description = {};
        if(strcmp(start,'%%'))
            title = rest;
            counter = 1;
            while(true)
                line = fulltext{counter+1};
                if(~isempty(line) && strcmp(line(1),'%'))
                    counter = counter + 1;
                    description = [description;strtrim(line(2:end))];       %#ok
                else
                    break;
                end
            end
        end
        evalCode = isempty(cell2mat(strfind(fulltext,doNotEvalTag)));
        fclose(fid);
       
        [funcsUsed,classesUsed] = dependsOn(which(mfile),PMTKroot());
        funcsUsed = setdiff(funcsUsed,excludeList);
    end

    function publishFile(mfile,outputDir,evalCode)
    % Publish an m-file to the specified output directory.
        
        options.evalCode = evalCode;
        options.outputDir = outputDir;
        options.format = 'html';
        options.createThumbnail = false;
        publish(mfile,options);
        evalin('base','clear all');
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
                info = [info, dirinfo([directory,'\',dirname])];        %#ok
            end
        end
    end

    function createViews(viewInfo)
    % Create all of the views, (i.e. root html files)
        createView1(viewInfo);
    end
    
    
    function createView1(viewInfo)
    % Create a view showing an alphabetical list of all of the demos complete 
    % with their titles and the classes and functions they use. 
        [sorted,perm] = sort(cellfun(@(str)lower(str),{viewInfo.functionName},'UniformOutput',false)); %#ok
        sortedInfo = viewInfo(perm);
        fid = setupHTMLfile(viewName1);
        setupTable(fid,{'Function Name','Title','Classes Used','Functions Used'},[20,40,20,20]);
        
        for i=1:numel(sortedInfo)
            title = sortedInfo(i).title;
            if(isempty(title)),title = '&nbsp;'; end    % empty html cell
            hprintf = @(txt)fprintf(fid,'\t<td> %s               </td>\n',txt);
            lprintf = @(link,name)fprintf(fid,'\t<td> <a href="%s"> %s </td>\n',link,name);
            fprintf(fid,'<tr bgcolor="white" align="left">\n');  
                %fprintf(fid,'\t<td> <a href="%s"> %s </td>\n',sortedInfo(i).htmlLink,sortedInfo(i).functionName);
                lprintf(sortedInfo(i).htmlLink,sortedInfo(i).functionName);
                hprintf(title);
                hprintf(prepareUsedList(sortedInfo(i).classesUsed));
                hprintf(prepareUsedList(sortedInfo(i).functionsUsed));
            fprintf(fid,'</tr>\n');  
        end
 
        fprintf(fid,'</table>');
        closeHTMLfile(fid);
    end

    function fid = setupHTMLfile(fname)
    % Setup a root HTML file    
        cdDocBase();
        d = date;
        fid = fopen(fname,'w+');
        fprintf(fid,'<html>\n');
        fprintf(fid,'<head>\n');
        fprintf(fid,'<font align="left" style="color:#990000"><h2>PMTK Documentation</h2></font>\n');
        fprintf(fid,'<br>Revision Date: %s<br>\n',d);
        fprintf(fid,'</head>\n');
        fprintf(fid,'<body>\n\n');
        fprintf(fid,'<br>\n');
    end
    
    function closeHTMLfile(fid)
    % Close a root HTML file    
        fprintf(fid,'\n</body>\n');
        fprintf(fid,'</html>\n');
        fclose(fid);
    end
    
    function setupTable(fid,names,widths)
    % Setup an HTML table with the specified field names and widths in percentages    
         fprintf(fid,'<table width="100%%" border="3" cellpadding="5" cellspacing="2" >\n');
         fprintf(fid,'<tr bgcolor="#990000" align="center">\n');
         for i=1:numel(names)
             fprintf(fid,'\t<th width="%d%%">%s</th>\n',widths(i),names{i});
         end
         fprintf(fid,'</tr>\n');
    end
   
    function charstr = prepareUsedList(cellstring)
    % Format the list of used classes or functions for placement in the html
    % table
        n = numel(cellstring);
        if(n>0)
            charstr = pubAndLink(cellstring{1});
            for k=2:numel(cellstring)
                charstr = [charstr,', ',pubAndLink(cellstring{k})];         %#ok
            end
        else
           charstr = '&nbsp;'; 
        end
    end
    
    
    function htmlString = pubAndLink(mfile)
    % Publish the specified file and return an html link to it.    
       outputDir = fullfile(destination,destRoot,'additional');
       cdDocBase();
       link = ['./additional/',mfile,'.html'];
       if(~exist(link,'file'))
            if(~makeRootOnly)
                publishFile(mfile,outputDir,false);
            end
       end
       link = ['./additional/',mfile,'.html'];
       htmlString = sprintf('<a href="%s">%s\n',link,mfile);
    end

    function cdDocBase()
    % Change directory to this documentation's root directory    
        cd(fullfile(destination,destRoot));
    end
    
   
    
    

end