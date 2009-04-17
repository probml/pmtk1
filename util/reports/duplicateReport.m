function duplicates = duplicateReport()
    
    [info,mfiles] = mfilelist(PMTKroot());          %#ok
    duplicates = {};
    for i=1:numel(mfiles)
        
       w = which('-all',mfiles{i});
       if(numel(w) == 2 && isequal(getText(w{1}),getText(w{2})))
           
           duplicates = [duplicates;mfiles{i}];
           which('-all',mfiles{i})
           
       end
        
    end
    
    
end