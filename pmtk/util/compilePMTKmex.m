function compilePMTKmex()
% Compile (i.e. mexify) every PMTK function with a #PMTKmex tag and a
% corresponding c-function. Also try and install lightspeed and fastfit. 
    savedDir = pwd;
    mfiles = findAllFilesWithTag('#PMTKmex');
    mfiles(ismember(mfiles,'compilePMTKmex.m')) = [];
    mfiles(ismember(mfiles,'removePMTKmex.m' )) = [];
    for i=1:numel(mfiles)
        file = mfiles{i};
        cfileName = [file(1:end-1),'c'];
        if(exist(cfileName,'file'))
            try
                cd(fileparts(which(file)));
                fprintf('Compiling %s\n',cfileName);
                mex(cfileName);
            catch
                fprintf('Could not compile %s\n',cfileName);
                clear mex;
                pause(1);
                try
                    delete([cfileName(1:end-1),mexext()]);
                catch
                end
            end
        else
            fprintf('Could not find %s\n',cfileName);
        end
    end
    
    if(~ismac)
        try
            cd(fullfile(PMTKroot,'util','lightspeed2.2'));
            install_lightspeed;
        catch
            cd(savedDir);
            error('Could not compile lightspeed functions.');
        end

    end
    cd(savedDir);
end