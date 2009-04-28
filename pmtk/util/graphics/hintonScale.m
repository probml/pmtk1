function [] = hintonScale(X, W, varargin)
% A modified version of hintonDiagram that allows the user to specify two matrices
% X, W; where X determines the color and W determines the size
% The user can specify two optional arguments
% 'map'   which colormap to use
% 'scale' do we scale X to use all colors from the colormap? (default = true)
  [map, scale] = processArgs(varargin, '-map', 'Jet', '-scale', true);
  % an [m,3] color matrix
  C = colormap(map);
  [ncolors] = size(C,1);

  % Make all data fit in the needed range
  if(any(X(:) < 0)), X = X - min(X(:)); end;
  Xmin = min(X(:)); Xmax = max(X(:));

  if(scale)
    X = (ncolors-1)*(X - Xmin)./(Xmax - Xmin) + 1;
  else
    X = (X / Xmax)*(ncolors-1) + 1;
  end

  % Make all weights positive
  W = abs(W);
  Smax = max(W(:)); Smin = Smax / 100;

  % DEFINE BOX EDGES
  xn1 = [-1 -1 +1]*0.5;
  xn2 = [+1 +1 -1]*0.5;
  yn1 = [+1 -1 -1]*0.5;
  yn2 = [-1 +1 +1]*0.5;
  
  xn = [-1 -1 +1 +1 -1]*0.5;
  yn = [-1 +1 +1 -1 -1]*0.5;
  
  [S,R] = size(W);
  
  cla reset
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
  
  for i=1:S
    for j=1:R
      m = sqrt((abs(W(i,j)) - Smin) / Smax);
      m = max(Smin, min(m,Smax)*0.95);
      if real(m)
        fill(xn*m+j,yn*m+i,C(ceil(X(i,j)),:))
        plot(xn1*m+j,yn1*m+i,'w',xn2*m+j,yn2*m+i,'k')
      end
    end
  end
  
  plot([0 R R 0 0]+0.5,[0 0 S S 0]+0.5,'w');
  grid on

end