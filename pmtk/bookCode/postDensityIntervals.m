%Illustration of central vs high posterior density intervals. 
%Written by Matthew Dunham
function postDensityIntervals


func = @(x)normpdf(x,0,1) + normpdf(x,6,1);
domain = -4:0.001:10;
mainPlot;
shade(func,0,-1.5,7.5,'b');
set(gca,'XTick',[],'YTick',[]);
annotation(gcf,'textarrow',[0.1795 0.2454],[0.2971 0.1431],'TextEdgeColor','none','FontSize',16,'FontName','Courier New','String',{'\alpha/2'});
annotation(gcf,'textarrow',[0.8522 0.7863],[0.2971 0.1431],'TextEdgeColor','none','FontSize',16,'FontName','Courier New','String',{'\alpha/2'});
if doPrintPmtk, doPrintPmtkFigures('centralInterval'); end;

figure;
mainPlot;
shade(func,0,-1.5,1.5,'b');
shade(func,0,4.5,7.5,'b');
line([-4;10],[func(-1.5),func(-1.5)],'Color','b','LineWidth',2);
set(gca,'XTick',[],'YTick',func(-1.5),'YTickLabel','pMIN');
if doPrintPmtk, doPrintPmtkFigures('HDP'); end;


function mainPlot
    plot(domain,func(domain),'-r','LineWidth',2.5);
    axis([-4,10,0,0.5]);  
end


%Shade under the specified function between 'left' and 'right' end points and
%above 'lower'.
function shade(func,lower,left,right,color)
    hold on;
    res = left:0.1:right;
    x = repmat(res,2,1);
    y = [lower*ones(1,length(res)) ; func(res)-0.005];
    line(x,y,'Color',color);
end

end