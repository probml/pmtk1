% Plot a Dirichlet distribution in 3d over x1,x2,x3 for a specified alpha
% vector. alpha must be of size 1x3.
% Warning, for certain values of alpha, particularly alpha < 1, this
% function can take up to 5 - 10 minutes to run and generates very large
% figures. If you receive an out of memory message, try turning down the
% interpolation resolution. Written by Matthew Dunham
function plotDirichlet3d(alpha)
    verbose = 1;
    figSize = 350;
    scrsz = get(0,'ScreenSize');
    figure('Position',[(scrsz(3)-figSize)/2,(scrsz(4)-figSize)/2,figSize,figSize]); 
    if(nargin < 1 | alpha == [1 1 1]) % Do this the fast way
       if(verbose),display('Using quick method for alpha = [1 1 1]'),end;
       x = [0;0;1]; y = [0;1;0]; z = ones(3,1)./sqrt(2);
       patch(x,y,z,[0,0,0.5]);
       axis([0 1 0 1 0 1]);
       grid on;
       return;
    end
    %evalRes = 0:0.01:1;   % Dirichlet is evaluated on a 3d meshgrid with this resolution. 
    evalRes = 0:0.1:1;
    interpRes = evalRes;  % Interpolation resolution, sufficient for alpha > 1 
    if(min(alpha) < 1)
        interpRes = 0:0.0005:1; % Necessary for a descent looking image if alpha < 1
    end
    
    if(verbose),display('Creating Grid...'),end;
    [x y z] = meshgrid(evalRes,evalRes,evalRes);
    MU = normalize([x(:) y(:) z(:)],2);
    if(verbose),display('done'),end;
    if(verbose),display('Calculating Probabilities...'),end;
    p = dirpdf(MU,alpha);
    if(verbose),display('done'),end;
    if(verbose),display('Removing duplicates from first 2 dimensions...'),end;
    [uniq keepNdx tossNdx] = unique(MU(:,1:2),'rows');
    x = uniq(:,1); y = uniq(:,2);
    p = p(keepNdx,1);
    if(verbose),display('done'),end;
    if(verbose),display('Removing entries with 0 probability...'),end;
	MU = [x,y,p];
    MU = MU(p>0,:);
    x = MU(:,1); y = MU(:,2); p = MU(:,3);
    if(verbose),display('done'),end;
    clear z MU uniq keep toss;
    if(verbose),display('Interpolating over a 2D grid, this may take up to 10 minutes...'),end;
    [XI,YI] = meshgrid(interpRes,interpRes);
    ZI = griddata(x,y,p,XI,YI,'linear');
    if(verbose),display('done'),end;
    if(verbose),display('normalizing...'),end;
    tmp = ZI(:);
    norm = sum(tmp(~isnan(tmp)));
    ZI = ZI ./ norm;
    if(verbose),display('done'),end;
    clear x y p;
    if(verbose),display('Plotting...'),end;
    surf(XI,YI,ZI);
    shading interp;
    camlight right;
    lighting phong;
    set(gca,'XTick',0:0.5:1);
    set(gca,'YTick',0:0.5:1);
    if(verbose),display('done'),end;
    
    % Calc unnormalized Dirichlet Probabilities
    % MU is nx3, alpha is 1x3
    function p = dirpdf(MU,alpha)
        expon = repmat((alpha -1),size(MU,1),1);
        p = prod(MU.^expon,2);
        p(isinf(p)) = 0;  
    end
end