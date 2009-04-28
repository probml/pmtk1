classdef DataTable
  % Wraps up X and Y into a single structure.
  % Can optionally assign names to the columns.
  % Can access subsets of the X and Y data together:
  % D = DataTable(X,Y);
  % D(1:n).X or D(1:n).Y extracts first n cases of X and Y
  % Concatenating D's is like concatentating the X's and Y's.
  
  %{
 Examples
  
  setSeed(0);
   X = rand(3,2)
  Y = rand(3,1)
  D = DataTable(X,Y)
  [ncases(D), ndimensions(D), noutputs(D)] % [3 2 1]
  
  X12 = D(1:2).X
   DH=[D D]
   DV = [D; D]
   DV2 = [D; DataTable(X(1,:),[])]
  
  X   =
    0.5488    0.5449
    0.7152    0.4237
    0.6028    0.6459
Y =
    0.4376
    0.8918
    0.9637
D = 
DataTable
properties:
         X: [3x2 double]
         Y: [3x1 double]
    Xnames: {'X1'  'X2'}
    Ynames: {'Y1'} 
X12 =
    0.5488    0.5449
    0.7152    0.4237
 DH = 
DataTable
properties:
         X: [3x4 double]
         Y: [3x2 double]
    Xnames: {'X1'  'X2'  'X1'  'X2'}
    Ynames: {'Y1'  'Y1'}
  
  DV = 
DataTable
properties:
         X: [6x2 double]
         Y: [6x1 double]
    Xnames: {'X1'  'X2'}
    Ynames: {'Y1'}
  
  DV2 = 
DataTable
properties:
         X: [4x2 double]
         Y: [3x1 double]
    Xnames: {'X1'  'X2'}
    Ynames: {'Y1'}
  
  %}
  
  properties
    X;
    Y;
    Xnames;
    Ynames;
  end
  
  methods
    function obj = DataTable(X, Y, Xnames, Ynames)
      if nargin ==0; return;end
      if nargin < 1
        X = [];
      end
      if nargin < 2
        Y = [];
      end
      if nargin < 3
        [n d] = size(X);
        Xnames = {};
        for j=1:d
          Xnames{j} = sprintf('X%d', j);
        end
      end
      if(nargin < 4)
        [n  T] = size(Y);
        Ynames = {};
        for j=1:T
          Ynames{j} = sprintf('Y%d',j);
        end
      end
      obj.X = X;
      obj.Y = Y;
      obj.Xnames = Xnames;
      obj.Ynames = Ynames;
    end
    
    function n = ndimensions(D)
      n = size(D.X,2);
    end
    
    function n = noutputs(D)
      n = size(D.Y,2);
    end
    
    function n = ncases(D)
      n = size(D.X,1);
    end
    
    function C = horzcat(A,B)
      C = DataTable([A.X B.X],[A.Y B.Y], [A.Xnames, B.Xnames], [A.Ynames, B.Ynames]);
      %C = [A;B];
    end
    
    function C = vertcat(A,B)
      if ~isequal(A.Xnames, B.Xnames) || ~isequal(A.Ynames, B.Ynames)
        warning('columns have different names');
      end
      C = DataTable([A.X;B.X],[A.Y;B.Y], A.Xnames, A.Ynames);
    end
    
    function obj = subsasgn(obj, S, value)
      if(numel(S) > 1)   % We have d(1:3,2).X = rand(3,1)  or d.X(1:3,2) = rand(3,1)
        colNDX = ':';
        if(strcmp(S(1).type,'.') && strcmp(S(2).type,'()')) %d.X(1:3,2) = rand(3,1)
          property = S(1).subs;
          rowNDX = S(2).subs{1};
          if(numel(S(2).subs) == 2)
            colNDX = S(2).subs{2};
          end
        elseif(strcmp(S(1).type,'()')&& strcmp(S(2).type,'.'))  %d(1:3,2).X = rand(3,1)
          property = S(2).subs;
          rowNDX = S(1).subs{1};
          if(numel(S(1).subs) == 2)
            colNDX = S(1).subs{2};
          end
        end
        switch property
          case 'X'
            obj.X(rowNDX,colNDX) = value;
          case 'Y'
            obj.Y(rowNDX,colNDX) = value;
          otherwise
            error([property, ' is not a property of this class']);
        end
      else
        switch S.type
          case {'()','{}'} %Parellel assignment to both X and y (value must be a cell array)
            obj.X(S.subs{:}) = value{1};
            obj.Y(S.subs{1},:) = value{2};
          case '.' %Still support full overwrite as in d.X = rand(10,10);
            obj = builtin('subsasgn', obj, S, value);
        end
      end
    end
    
    function B = subsref(A, S)
      if(numel(S) > 1)  % We have d(1:3,:).X for example or d.X(1:3,:)
        colNDX = ':';
        if(strcmp(S(1).type,'.') && strcmp(S(2).type,'()')) %d.X(1:3,:)
          property = S(1).subs;
          rowNDX = S(2).subs{1};
          if(numel(S(2).subs) == 2)
            colNDX = S(2).subs{2};
          end
        elseif(strcmp(S(1).type,'()')&& strcmp(S(2).type,'.'))  %d(1:3,:).X
          property = S(2).subs;
          rowNDX = S(1).subs{1};
          if(numel(S(1).subs) == 2)
            colNDX = S(1).subs{2};
          end
        end
        switch property
          case 'X'
            B = A.X(rowNDX,colNDX);
          case 'Y'
            B = A.Y(rowNDX,:);
          otherwise
            error([property, ' is not a property of this class']);
        end
        return;
      end
      % numel(S)=1
      switch S.type    %d(1:3,:)   for example
        case {'()'}
          y = [];
          Xnames = {};
          Ynames = A.Ynames;
          colNDX = ':';
          if(numel(S.subs) == 2)
            colNDX = S.subs{2};
          end
          if(~isempty(A.Y))
            Y = A.Y(S.subs{1},:);
          end
          if(~isempty(A.Xnames))
            Xnames = A.Xnames(colNDX);
          end
          B = DataTable(A.X(S.subs{1},colNDX),Y,Xnames,Ynames);
        case '.' %Still provide access of the form d.X and d.y
          B = builtin('subsref', A, S);
      end
    end

  end
  
 
  
end
  
