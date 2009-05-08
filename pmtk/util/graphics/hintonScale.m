function [] = hintonScale(varargin)
% A modified version of hintonDiagram that allows the user to specify two matrices
% X, W; where X determines the color and W determines the size
% The user can specify two optional arguments
% 'map'   which colormap to use
% We make use of rel2absX and rel2absY from pmtk/bookCode
  if nargin < 2
    X = varargin{1}{1}; map = 'Jet';
    if(numel(varargin{1}) == 1), W = NaN*ones(size(X)); end;
    nplots = 1;
    Xmin = min(X(:)); Xmax = max(X(:));
    Smin = min(W(:))*0.95; Smax = max(W(:))*1.05;
  end
  %[map] = processArgs(varargin, '-map', 'Jet');
  if(nargin > 2)
    nplots = nargin / 2; if(round(nplots) ~= nplots), nplots = 1; end;
    localMinX = zeros(nplots,1); localMaxX = zeros(nplots,1);
    localMinW = zeros(nplots,1); localMaxW = zeros(nplots,1);
    for i=1:nplots
      [imap, ititle] = processArgs(varargin{2*i}, '-map', 'Jet', '-title', '');
      map{i,:} = imap; title{i} = ititle;
      allX{i} = varargin{2*i-1}{1};
      localMinX(i) = min(min(allX{i})); localMaxX(i) = max(max(allX{i}));
      if(numel(varargin{2*i-1}) > 1)
        allW{i} = abs(varargin{2*i-1}{1});
      else
        allW{i} = NaN*ones(size(varargin{2*i-1}{1}));
      end
      localMinW(i) = min(min(allW{i})); localMaxW(i) = max(max(allW{i}));
    end
    Xmin = min(localMinX); Xmax = max(localMaxX);
    Smin = min(localMinW)*0.95; Smax = max(localMaxW)*1.05;
  end


  if(size(map,1) > 1)
  allSameMap = all(strcmpi(map{1}, map));
    if(~allSameMap), warning('Different maps in subplots not supported yet.  Using the first'); end;
  map = map{1};
  end
  C = colormap(map);
  [ncolors] = size(C,1);
  transform = @(x)(round(ncolors*(x - Xmin + 1/2)./(Xmax - Xmin + 1/2)));


  figure();

  [plotRows,plotCols] = nsubplots(nplots);
  for p=1:nplots
    subplot(plotRows, plotCols, p);
    X = allX{p}; W = allW{p};
    % an [m,3] color matrix


    % Make all data fit in the needed range
    %if(any(X(:) < 0)), X = abs(X); end;
    %Xmin = min(X(:)); Xmax = max(X(:));
    
    % Make all weights positive
    %W = abs(W);
    %Smax = max(abs(W(:))); Smin = Smax / 100;

    % DEFINE BOX EDGES
    xn1 = [-1 -1 +1]*0.5;
    xn2 = [+1 +1 -1]*0.5;
    yn1 = [+1 -1 -1]*0.5;
    yn2 = [-1 +1 +1]*0.5;
    
    xn = [-1 -1 +1 +1 -1]*0.5;
    yn = [-1 +1 +1 -1 -1]*0.5;
    
    [S,R] = size(W);
    
    %cla reset
    hold on
    set(gca,'xlim',[0 R]+0.5);
    set(gca,'ylim',[0 S]+0.5);
    set(gca,'xlimmode','manual');
    set(gca,'ylimmode','manual');
    xticks = get(gca,'xtick');
    set(gca,'xtick',xticks(find(xticks == floor(xticks))))
    yticks = get(gca,'ytick');
    set(gca,'ytick',yticks(find(yticks == floor(yticks))))
    set(gca,'ydir','reverse');
    
    m = ((abs(W) - Smin) / (Smax - Smin));
    for i=1:S
      for j=1:R
        if real(m(i,j))
          fill(xn*m(i,j)+j,yn*m(i,j)+i,C(transform(X(i,j)),:));
          plot(xn1*m(i,j)+j,yn1*m(i,j)+i,'w',xn2*m(i,j)+j,yn2*m(i,j)+i,'k')
        end
      end
    end
    
    plot([0 R R 0 0]+0.5,[0 0 S S 0]+0.5,'w');
    grid on;

  end
    msize = @(m)((abs(m) - Smin) / (Smax - Smin));
    breaks = 0.05:0.20:0.85;
    breaksX = (0.05:0.20:0.85)/R;
    breaksY = (0.05:0.20:0.85)/S;
    yloc = 0.05 + cumsum(0.15*ones(numel(breaksX),1));
    plotSize = get(gca,'Position');
    scaleY = plotSize(4);
    scaleX = plotSize(3);
    for s=1:length(breaksX)
       location = [0.01, yloc(s), breaksX(s)*scaleX, breaksY(s)*scaleY];
       annotation('rectangle', location, 'color', 'k');
       annotation('textbox', location + [-0.01, 0.05, 0, 0], 'String', sprintf('%3.2f', breaks(s)*(Smax - Smin) + Smin), 'LineStyle', 'none');
    end
    annotation('textbox', [0.00, 0.99, 0.20, 0], 'String', 'Abs(Weight)', 'LineStyle', 'none');
    annotation('textbox', [0.90, 0.99, 0.20, 0], 'String', 'Value', 'LineStyle', 'none');  
    axes('Position', [0.05 0.05 0.95 0.85], 'Visible', 'off');
    caxis([Xmin, Xmax]);
    colorbar;
end