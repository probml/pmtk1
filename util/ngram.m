% Compute and plot unigrams and bigrams of Darwin's "On the origin of
% species". Written by Matthew Dunham

close all;
verbose = 1;

if(1) % 1 for precomputed uni and bigrams, 0 to recompute. 
    load ngramData;
else
    fid = fopen('darwin.txt'); 
    if(verbose),display('reading file...'),end;
    data = fread(fid); 
    if(verbose),display('done'),end;
    fclose(fid); 
    
    lcase = abs('a'):abs('z');
    ucase = abs('A'):abs('Z');
    caseDiff = abs('a') - abs('A');
    
    if(verbose),display('converting all letters to lower case...'),end;
    caps = ismember(data,ucase);
    data(caps) = data(caps)+caseDiff;
    if(verbose),display('done'),end;
    
    if(verbose),display('removing punctuation...'),end;
    validSet = [abs(' ') lcase];
    data = data(ismember(data,validSet));
    if(verbose)
        display('done');
        display('computing unigrams and bigrams...');
    end
    ugrams = zeros(27,1);
    bigrams = zeros(27,27);
    shiftVal = abs('a') - 2;           % 'a' will be at index 2
    shift = @(x) max(1,x - shiftVal);  % space will be at index 1
    for i=1:length(data)-1
        fromIDX = shift(data(i));
        ugrams(fromIDX) = ugrams(fromIDX) + 1;
        toIDX = shift(data(i+1));
        bigrams(fromIDX,toIDX) = bigrams(fromIDX,toIDX) + 1;
    end
    if(verbose)
        display('done');
        display('removing extra whitespace...');
        display('done');
    end
    ugrams(1) = ugrams(1)-bigrams(1,1);
    bigrams(1,1)=0; %space to space
    last = shift(data(length(data)));
    ugrams(last) = ugrams(last) + 1;
    ugramsNorm = mkStochastic(ugrams);
    bigramsNorm = mkStochastic(bigrams);
    clear ans caps fid loadExisting i last fromIDX toIDX;
    save ngramData;
end


if(1) % Plot unigram and bigram frequencies
    main = figure;
    hintonDiagram(ugrams);
    title('Unigrams');
    uniAx = gca;
    set(uniAx,'XTick',[],'YTick',1:27,'Color','k','Position',[0.25,0.1,0.04,0.8],'FontName','Courier');
    set(findobj(uniAx,'Type','Patch'),'FaceColor','w')
    set(findobj(uniAx,'Type','line'),'Color','k');
    xlabel('');  ylabel('');  grid off;
    letters = {'_','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'};
    spaces = repmat('   ',27,1);
    labels = [num2str((1:27)') spaces num2str(ugramsNorm,'%4.5f') spaces char(letters)];
    set(uniAx,'YTickLabel',labels);
    tmp = figure;
    hintonDiagram(bigrams);
    title('Bigrams');
    xlabel(''); ylabel('');
    biAx = gca;
    set(biAx,'Parent',main,'Position',[0.4,0.1,0.55,0.8]);
    close(tmp);
    set(biAx,'XTick',1:27,'YTick',1:27,'Color','k','XAxisLocation','top','XTickLabel',letters,'YTickLabel',letters,'FontName','Courier');
    set(findobj(biAx,'Type','Patch'),'FaceColor','w')
    set(findobj(biAx,'Type','line'),'Color','k');
    grid off;
    set(main,'Color','w');
end

if(0) % Print a table of unigram frequencies to the console
    fprintf('\n%10s %7s %12s\n\n','Index','Letter','Frequency');
    for i = 1:length(validSet)
        fprintf('%8s %6s %15s\n',num2str(i),char(validSet(i)),num2str(ugramsNorm(i),'%4.3f'));
    end
end


if(0) % Print a histogram of unigram frequencies
    ugramHist = figure;
    axes2 = axes('Parent',ugramHist,'XTickLabel',letters,'XTick',1:27);
    hold on;
    bar(ugramsNorm);
    title('Unigrams');
end