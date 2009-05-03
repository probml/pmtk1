function objectCreationTest()
% Try instantiating every class.     
    nprobs = 0;
    excludeList = {'Graphlayout','GraphlayoutNode'};
    classes = setdiff(getClasses(PMTKroot()),excludeList);
    fprintf('will try to instantiate %d classes\n', length(classes));
    for i=1:numel(classes)
        if ~isabstract(classes{i})
           try
              feval(classes{i}); 
           catch ME %#ok
              fprintf('Could not instantiate %s with 0 arguments\n',classes{i});
              nprobs = nprobs + 1;
           end
        end
    end
    assert(nprobs == 0);
end