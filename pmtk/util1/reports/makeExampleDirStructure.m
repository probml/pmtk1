cd c:\PMTK

classes = getClasses('.');

cd './examples'
for i=1:numel(classes);
   
      system(['mkdir ',classes{i},'Examples']);
  
 
end