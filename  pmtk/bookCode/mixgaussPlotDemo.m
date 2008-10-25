%Display plots of an arbitrary 2d pdf using a mixture of 3 2D Gaussians
%Written by Matthew Dunham
% Based on Bishop fig 2.23

function mixgaussPlotDemo

pi = [0.5 0.3 0.2]; % Mixing weights

u1 = [0.22 0.45];
u2 = [0.5 0.5];
u3 = [0.77 0.55];

s1 = [0.018  0.01 ;  0.01 0.011];
s2 = [0.011 -0.01 ; -0.01 0.018];
s3 = s1;

plotA = 1;      % Contours of the individual pdfs
plotB = 2;      % Contour of the full pdf
surface = 3;    % Surf plot of the full pdf

figure(plotA);
hold on;

contourPlot(@(x)mvnpdf(x,u1,s1),[4,11,15],'r',0);
contourPlot(@(x)mvnpdf(x,u2,s2),[4,11,15],'g',0);
contourPlot(@(x)mvnpdf(x,u3,s3),[4,11,15],'b',0);

figure(plotB);
pdf = @(X)pi(1)*mvnpdf(X,u1,s1) + pi(2)*mvnpdf(X,u2,s2) + pi(3)*mvnpdf(X,u3,s3);
contourPlot(pdf,1:7,'r',1);


%%
function contourPlot(func,contours,colour,surfToo)
       stepSize = 0.005;
       [x,y] = meshgrid(-0.2:stepSize:1.2,-0.2:stepSize:1.2);
       [r,c]=size(x);
       data = [x(:) y(:)];
       p = func(data);
       p = reshape(p, r, c);
       contour(x,y,p,colour,'LineWidth',2,'LevelList',contours);
       set(gca,'XTick',[0,0.5,1]);
       set(gca,'YTick',[0,0.5,1]);
       if(surfToo)
           figure(surface);
           brown = [0.8 0.4 0.2];
           surf(x,y,p,'FaceColor',brown,'EdgeColor','none');
           hold on;
           view([-27.5 30]);
           camlight right;
           lighting phong;
           axis off;
       end


end


annotation(plotA,'textbox','String',{'0.2'},'FontSize',14,...
   'FontName','Arial',...
   'FitHeightToText','off',...
   'LineStyle','none',...
   'Position',[0.7462 0.4452 0.05842 0.03737]);

annotation(plotA,'textbox','String',{'0.5'},'FontSize',14,...
   'FontName','Arial',...
   'FitHeightToText','off',...
   'LineStyle','none',...
   'Position',[0.2582 0.3119 0.05842 0.03737]);

annotation(plotA,'textbox','String',{'0.3'},'FontSize',14,...
   'FontName','Arial',...
   'FitHeightToText','off',...
   'LineStyle','none',...
   'Position',[0.5762 0.306 0.05842 0.03737]);

if 0
annotation(plotA,'textbox','String',{'(a)'},'FontSize',14,...
   'FontName','Arial',...
   'FitHeightToText','off',...
   'LineStyle','none',...
   'Position',[0.2201 0.7771 0.05007 0.04897]);

annotation(plotB,'textbox','String',{'(b)'},'FontSize',14,...
   'FontName','Arial',...
   'FitHeightToText','off',...
   'LineStyle','none',...
   'Position',[0.2201 0.7771 0.05007 0.04897]);

annotation(surface,'textbox','String',{'(c)'},'FontSize',14,...
   'FontName','Arial',...
   'FitHeightToText','off',...
   'LineStyle','none',...
   'Position',[0.248 0.4201 0.05007 0.04897]);
end



end
