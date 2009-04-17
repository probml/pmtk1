function emailList = parseDownloads(filePath)
% Parse the PMTKdownloads.txt file and extract the unique e-mail addresses
% found there. 
    email = cellfuncell(@(line)subc(tokenize(line,','),5),getText(filePath));
    emailList = unique(email(cellfun(@(c)~isempty(c)&&c==1,regexpi(email,'\w*[\w\.]+@\w+[\w\.]*\.\w+'))));
end