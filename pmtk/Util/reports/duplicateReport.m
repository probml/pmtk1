function duplicates = duplicateReport()
    
    [info,mfiles] = mfilelist(PMTKroot());          %#ok
    duplicates = {};
    for i=1:numel(mfiles)
        
       w = which('-all',mfiles{i});
       if(numel(w) > 1)
           duplicates = [duplicates;mfiles{i}];
       end
        
    end
    
    
end