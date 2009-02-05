function objectCreationTest()
    
    nprobs = 0;
    excludeList = {'Graphlayout','GraphlayoutNode'};
    classes = setdiff(getClasses(PMTKroot()),excludeList);
    for i=1:numel(classes)
        if ~isabstract(classes{i})
           try
              feval(classes{i}); 
           catch
              fprintf('Could not instantiate %s with 0 arguments\n',classes{i});
              nprobs = nprobs + 1;
           end
        end
    end
    assert(nprobs == 0);
    
    
end