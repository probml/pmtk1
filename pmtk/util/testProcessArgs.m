function testProcessArgs()
% This function tests processArgs    
    
    
    %% All positional
    [a,b,c,d,e] = test1();                          assert(a==1 && b==2 && c==3 && d==4 && e==5);
    [a,b,c,d,e] = test1(5,4,3,2,1);                 assert(a==5 && b==4 && c==3 && d==2 && e==1);
    [a,b,c,d,e] = test1(5,5,5);                     assert(a==5 && b==5 && c==5 && d==4 && e==5);
    [a,b,c,d,e] = test1(10);                        assert(a==10 && b==2 && c==3 && d==4 && e==5);
    [a,b,c,d,e] = test1(10,[],[],[],10);            assert(a==10 && b==2 && c==3 && d==4 && e==10);
    [a,b,c,d,e] = test1(10,{},{},[],10);            assert(a==10 && b==2 && c==3 && d==4 && e==10);
    [a,b,c,d,e] = test1(10,{},20,[],[]);            assert(a==10 && b==2 && c==20 && d==4 && e==5);
    [a,b,c,d,e] = test1([]);                        assert(a==1 && b==2 && c==3 && d==4 && e==5);
    [a,b,c,d,e] = test1([],[],[],[],[]);            assert(a==1 && b==2 && c==3 && d==4 && e==5);
    %% Named
    [a,b,c,d,e] = test1('-first',5,'-fifth',1);                                    assert(a==5 && b==2 && c==3 && d==4 && e==1);
    [a,b,c,d,e] = test1('-first',5,'-second',4,'-third',3,'-fourth',2,'-fifth',1); assert(a==5 && b==4 && c==3 && d==2 && e==1);
    [a,b,c,d,e] = test1('-first',[],'-second',4,'-third','','-fourth',2);          assert(isempty(a) && b==4 && isempty(c) && d==2 && e==5);
    [a,b,c,d,e] = test1('-fourth',20,'-third',20);                                 assert(a==1 && b==2 && c==20 && d==20 && e==5);

    %% Mixed
    [a,b,c,d,e] = test1(17,22,'-fourth',20);                                       assert(a==17 && b==22 && c==3 && d==20 && e==5);
    [a,b,c,d,e] = test1(17,22,'-third',10,'-fifth',10);                            assert(a==17 && b==22 && c==10 && d==4 && e==10);
    [a,b,c,d,e] = test1([],22,'-third',[],'-fifth',10);                            assert(a==1 && b==22 && isempty(c) && d==4 && e==10);
    [a,b,c,d,e] = test1(1,[],[],'-fifth',10);                                      assert(a==1 && b==2 && c==3 && d==4 && e==10);
    %% Error Conditions
    caught = false;
    try
        test1(1,2,3,4,5,6)
    catch ME
        caught = true;
    end
    assert(caught);
    
    caught = false;
    try
        test1('-first',1,2,3)
    catch
        caught = true;
    end
    assert(caught);
    
    caught = false;
    try
       test1('-ffirst',1) 
    catch   
        caught = true;
    end
    assert(caught);
    
    caught = false;
    try
        test1('-first','-second',2)
    catch
        caught = true;
    end
    assert(caught);
    
    caught = false;
    try
        test1('-first','test',2,3)
    catch
        caught = true;
    end
    assert(caught);
    
    caught = false;
    try
        test1('-first','-second')
    catch
        caught = true;
    end
    assert(caught);
    
    caught = false;
    try
        test1(1,2,'-first',3)
    catch
        caught = true;
    end
    assert(caught);
    
    caught = false;
    try
        test1('-first',1,'-first',2)
    catch 
        caught = true;
    end
    assert(caught);
    
    caught = false;
    
%% required args 
    
    [a,b,c,d,e] = test2('-first',5,'-third',1);             assert(a==5 && b==2 && c==1 && d==4 && e==5);
    [a,b,c,d,e] = test2(5,'-third',1);                      assert(a==5 && b==2 && c==1 && d==4 && e==5);
    [a,b,c,d,e] = test2(5,[],1);                            assert(a==5 && b==2 && c==1 && d==4 && e==5);
    [a,b,c,d,e] = test2(5,99,1);                            assert(a==5 && b==99 && c==1 && d==4 && e==5);
    [a,b,c,d,e] = test2(5,[],'-third',[]);                  assert(a==5 && b==2 && isempty(c) && d==4 && e==5);
    
%% require args error conditions    
    
    caught = false;
    try
        [a,b,c,d,e] = test2('-first',5);             
    catch
        caught = true;
    end
    assert(caught);
    caught = false;
    
    caught = false;            
    try
        [a,b,c,d,e] = test2([],3,[]);
    catch
        caught = true;
    end
    assert(caught);
    caught = false;
    
%% type checking    
    

    [a,b,c,d,e] = test3();                                                 assert(a==1 && b==2 && isequal(d,{4}) && strcmp(e,'foo'));
    [a,b,c,d,e] = test3(99,100,ProbDist(),{44},'test');                    assert(a==99 && b==100 && isequal(d,{44}) && strcmp(e,'test'));    
    [a,b,c,d,e] = test3(99,100,MvnDist(),{'no'},' ');                      assert(a==99 && b==100 && isequal(d,{'no'}) && strcmp(e,' '));    
    
    
    
%% type checking error conditions    
     
    caught = false;            
    try
      [a,b,c,d,e] = test3(int32(5));  
    catch
        caught = true;
    end
    assert(caught);
    caught = false;
    
    caught = false;              
    try
      [a,b,c,d,e] = test3(1,2,DiscreteDist(),4,5);  
    catch
        caught = true;
    end
    assert(caught);
    caught = false;


%%  both type checking and required args

    [a,b,c,d,e] = test4('-first',5,'-third',MvnDist(),'-fourth',{2},'-second',10); assert(a==5 && b==10 && isa(c,'MvnDist'),isequal(d,{2}));
    [a,b,c,d,e] = test4(5,'-third',MvnDist(),'-fourth',{2},'-second',10);          assert(a==5 && b==10 && isa(c,'MvnDist'),isequal(d,{2}));
    [a,b,c,d,e] = test4(5,10,MvnDist(),'-fourth',{2});                             assert(a==5 && b==10 && isa(c,'MvnDist'),isequal(d,{2}));
    [a,b,c,d,e] = test4(5,10,MvnDist(),'-fourth',{});                              assert(a==5 && b==10 && isa(c,'MvnDist'),isequal(d,{2}));
    [a,b,c,d,e] = test4(5,10,MvnDist(),'-fourth',{},'-fifth','fine');              assert(a==5 && b==10 && isa(c,'MvnDist'),isequal(d,{2}));
    
%% both type checking and required args error conditions    
    
    caught = false;              
    try
        [a,b,c,d,e] = test4(1,2,DiscreteDist(),{},'foo');  
    catch
        caught = true;
    end
    assert(caught);
    caught = false;
    
    caught = false;              
    try
        [a,b,c,d,e] = test4(1,2,'test',{},'foo');  
    catch
        caught = true;
    end
    assert(caught);
    caught = false;

    caught = false;              
    try
        [a,b,c,d,e] = test4(1,2,'test',{3},'foo','-test',3);  
    catch
        caught = true;
    end
    assert(caught);
    caught = false;
    
    caught = false;                
    try
        [a,b,c,d,e] = test4(1,2);  
    catch
        caught = true;
    end
    assert(caught);
    caught = false;

    

    function [a,b,c,d,e] = test1(varargin)
    % test standard usage    
       [a,b,c,d,e] = processArgs(varargin,'-first',1,'-second',2,'-third',3,'-fourth',4,'-fifth',5);
    end
    
    function [a,b,c,d,e] =  test2(varargin)
    % test required args
        [a,b,c,d,e] = processArgs(varargin,'-*first',1,'-second',2,'-*third',3,'-fourth',4,'-fifth',5);
    end
    
    function [a,b,c,d,e] = test3(varargin)
    % test type checking   
        [a,b,c,d,e] = processArgs(varargin,'+-first',1,'-second',2,'+-third',ProbDist(),'+-fourth',{4},'+-fifth','foo');
    end
    
    
    function [a,b,c,d,e] = test4(varargin)
    % both type checking and required args
        [a,b,c,d,e] = processArgs(varargin,'*+-first',1,'*-second',2,'+*-third',ProbDist(),'*+-fourth',{4},'+-fifth','foo');
    end
    
    
    caught = false;                
    try
       test5(); 
    catch
        caught = true;
    end
    assert(caught);
    caught = false;

    caught = false;                
    try
       test6(); 
    catch
        caught = true;
    end
    assert(caught);
    caught = false;

    
    caught = false;                
    try
       test7(); 
    catch
        caught = true;
    end
    assert(caught);
    caught = false;
    
    
    caught = false;                
%     try
%        test8(); 
%     catch
%         caught = true;
%     end
%     assert(caught);
%     caught = false;
    
    caught = false;                
    try
       test9(); 
    catch
        caught = true;
    end
    assert(caught);
    caught = false;
    
    
    caught = false;                
    try
       test10(); 
    catch
        caught = true;
    end
    assert(caught);
    caught = false;
    
    caught = false;                
    try
       test11('-foo',1); 
    catch
        caught = true;
    end
    assert(caught);
    caught = false;
    
    
    function test5(varargin)
    % test programmer error conditions    
        [a,b,c] = processArgs(varargin,'-first',1,'-second',2,'-third',3,'-fourth',4,'-fifth',5);
    end
    
     
    function test6(varargin)
    % test programmer error conditions    
        [a,b,c,d,e] = processArgs(varargin,'first',1,'-second',2,'-third',3,'-fourth',4,'-fifth',5);
    end
    
     function test7(varargin)
    % test programmer error conditions    
        [a,b,c,d,e] = processArgs(varargin,'-first','-second',2,'-third',3,'-fourth',4,'-fifth',5);
    end
    
     function test8(varargin)
    % test programmer error conditions    
        [a,b,c,d,e] = processArgs(varargin,'-first',3,'-first',2,'-third',3,'-fourth',4,'-fifth',5);
     end
    
    function test9(varargin)
    % test programmer error conditions    
        [a,b,c,d,e] = processArgs(varargin,'',3,'-second',2,'-third',3,'-fourth',4,'-fifth',5);
    end
    
    function test10(varargin)
    % test programmer error conditions    
        [a,b,c,d,e] = processArgs(varargin);
    end
    
      
    function test11(varargin)
    % test programmer error conditions    
       [a] = processArgs('-foo',2);
    end
    
    
%% additional tests

    [a,b,c] = test12(22,33,44);          assert(a==22 && b==33 && c==44);
    [a,b,c] = test12(22,33,'-arg1',44);  assert(a==22 && b==33 && c==44);
    [a,b,c] = test12(22,33);             assert(a==22 && b==33 && c==22);
    function [a,b,c] =  test12(a,b,varargin)
       c = processArgs(varargin,'-arg1',22); 
    end
    
    
%% example
obj = '';
outerFunction(obj,'-first',1,'-second',MvnDist(),'-third',22,'-fourth',10); 
outerFunction(obj,'-fourth',3,'-second',MvnDist(),'-first',12);
outerFunction(obj,1,MvnDist(),3);
outerFunction(obj,1,MvnDist(),3,[]);                           
outerFunction(obj,'-first',1,'-second',DiscreteDist(),'-third',[]);
outerFunction(obj,1,MvnDist(),'-fourth',10);                           


function [a,b,c,d] = outerFunction(obj,varargin)
  [a,b,c,d] = processArgs(varargin,'*-first',[],'*+-second',ProbDist(),'-third',18,'+-fourth',23);
end
    
    
    
    
    
    
end