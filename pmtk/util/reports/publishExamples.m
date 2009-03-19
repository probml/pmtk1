function publishExamples(destination)
% This function automatically generates documentation for the PMTK system.
% By default the root doc directory is set to 'C:\kmurphy\pmtkLocal\doc'
% but you can specify another as a parameter to this function. Within this
% root directory, it creates a directory structure and html files
% corresponding to the examples in the example directory.


%%  Adjustable parameters
    excludeTags         = {'#broken','#inprogress','#doNotPublish'};  % do not include demos with these tags
    viewName1           = 'publishedExamples.html';                   % name of root HTML file for the docs
    defaultDocDir       = 'C:\kmurphy\pmtkLocal\doc';                 % default directory to store the documentation
    doNotEvalTag        = '%#doNotEval';                              % if this tag is present in the function's documentation, it is not evaluated when published
    excludeFnNameList   = {};                                         % functions listed here, or in trivialFunctionList.txt, are not displayed in the "Functions Used" column
    makeRootOnly        = false;                                      % if true, only the root html file is generated, nothing else is published. This can be useful when only small changes have been made. 
    globalEval          = true;                                       % if false, none of the examples are evaluated - only converted to html. 
%% 
    if(exist('trivialFunctionList.txt','file'))
        excludeFnNameList = [excludeFnNameList,getText('trivialFunctionList.txt')];
    end
    
    originalDirectory = pwd;                                    % save current directory
    if(nargin == 0), destination = defaultDocDir;  end          % this is where the docs will live
    cd(PMTKroot());                                             % change to the base PMTK directory
    destRoot = makeDestinationDirectory();                      % make the documentation directory

    %%
    % This struct array stores all of the information collected about the
    % demos. Currently the subfunction createView1() uses this info to
    % generate the main html file. To create a different view on the same 
    % data, simply write another subfunction, say createView2(), and add
    % the line createView2(viewInfo) to the createViews() subfunction. 
    viewInfo = struct('functionName',{},'title'       ,{} ,'description'  ,{},...
                      'htmlLink'    ,{},'classesUsed' ,{} ,'functionsUsed',{},...
                      'evalCode'    ,{},'outputDir'   ,{}); 
    %%                  
    allexamples = processExamples({},excludeTags,0,false);
    for ex=1:numel(allexamples)
        [viewInfo(ex).functionName                    ,...
                        viewInfo(ex).title            ,...
                        viewInfo(ex).description      ,...
                        viewInfo(ex).functionsUsed    ,...
                        viewInfo(ex).classesUsed      ,...
                        viewInfo(ex).evalCode]        =...
                             getDemoInfo(allexamples{ex});
        viewInfo(ex).htmlLink = ['./examples/',viewInfo(ex).functionName,'.html'];
        viewInfo(ex).outputDir = fullfile(destination,destRoot,'examples');  
    end
    
    if(not(makeRootOnly))
        pause(1);               % bugfix: wait for Matlab to release file locks
        for v=1:numel(viewInfo)
            evalin('base','clear all'); 
            publishFile(viewInfo(v).functionName,viewInfo(v).outputDir,viewInfo(v).evalCode); 
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
    
    function [fname,title,description,funcsUsed,classesUsed,evalCode] = getDemoInfo(mfile)
    % Collect information about the specified demo.
        fname = mfile(1:end-2);
        fulltext = getText(mfile);
        [start,rest] = strtok(fulltext{1},' ');
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
        evalCode = globalEval && not(tagsearch(mfile,doNotEvalTag));
        [funcsUsed,classesUsed] = dependsOn(which(mfile),PMTKroot());
        funcsUsed = setdiff(funcsUsed,excludeFnNameList);
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
        try
            cd(destination);   % See if it already exists, if not, create it  
        catch ME               %#ok
            err = system(['mkdir ',destination]);
            if(err)            % if it can't create it, error
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

    function createViews(viewInfo)
    % Create all of the views, (i.e. root html files)
        createView1(viewInfo);
    end
    
    function createView1(viewInfo)
    % Create a view showing an alphabetical list of all of the demos complete 
    % with their titles and the classes and functions they use. 
        [sorted,perm] = sort(cellfuncell(@(str)lower(str),{viewInfo.functionName})); 
        sortedInfo = viewInfo(perm);
        fid = setupHTMLfile(viewName1);
        setupTable(fid,{'Function Name','Title','Classes Used','Functions Used'},[20,40,20,20]);
        
        for i=1:numel(sortedInfo)
            title = sortedInfo(i).title;
            if(isempty(title)),title = '&nbsp;'; end    % empty html cell
            hprintf = @(txt)fprintf(fid,'\t<td> %s               </td>\n',txt);
            lprintf = @(link,name)fprintf(fid,'\t<td> <a href="%s"> %s </td>\n',link,name);
            fprintf(fid,'<tr bgcolor="white" align="left">\n');  
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
        fprintf(fid,'<br>Auto-generated by publishExamples.m<br>\n');
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
       outputDir = fullfile(destination,destRoot,'supportingFns');
       cdDocBase();
       link = ['./supportingFns/',mfile,'.html'];
       if(~exist(link,'file'))
            if(~makeRootOnly)
                publishFile(mfile,outputDir,false);
            end
       end
       link = ['./supportingFns/',mfile,'.html'];
       htmlString = sprintf('<a href="%s">%s\n',link,mfile);
    end

    function cdDocBase()
    % Change directory to this documentation's root directory    
        cd(fullfile(destination,destRoot));
    end

end