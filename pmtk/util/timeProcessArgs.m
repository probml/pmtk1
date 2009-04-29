function timeProcessArgs()
% compare running time of processArgs and process_options.
    
    tic
    for i=1:1000
        testPO('-a',1,'-k',11,'-c',3,'-e',5,'-f',6,'-d',4,'-g',7,'-h',8,'-i',9,'-b',2);
    end
    t = toc;
    fprintf('process_options took %f seconds\n',t);

    tic
    for i=1:1000
        testPA('-a',1,'-k',11,'-c',3,'-e',5,'-f',6,'-d',4,'-g',7,'-h',8,'-i',9,'-b',2);
    end
    t = toc;
    fprintf('processArgs took %f seconds\n',t);
    
    
    
    
    function [a,b,c,d,e,f,g,h,i,j,k] = testPA(varargin)
        [a,b,c,d,e,f,g,h,i,j,k] = processArgs   (varargin,'*-a',23,'-b','foo','-c',[],'-d',10,'-e',struct,'-f',{10},'-g',1,'-h',1,'-i','test','-j',10,'-k',10);
    end
    
    function [a,b,c,d,e,f,g,h,i,j,k] = testPO(varargin)
        [a,b,c,d,e,f,g,h,i,j,k] = process_options(varargin,'-a',23,'-b','foo','-c',[],'-d',10,'-e',struct,'-f',{10},'-g',1,'-h',1,'-i','test','-j',10,'-k',10);
    end
    
    
    
end