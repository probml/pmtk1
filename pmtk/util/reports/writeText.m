function writeText(text,fname)
% Write text to a file. 
%
% WARNING: Any existing text inside fname will be lost!
% 
% INPUTS:
%
% text  - an N-by-1 cell array of strings written out as N separate lines.
% fname - the name of the output file. 
%
    fid = fopen(fname,'w+');
    if fid < 0
       error('could not open %s',fname); 
    end
    for i=1:numel(text)
       fprintf(fid,'%s\n',text{i});
    end
    fclose(fid);
    
end