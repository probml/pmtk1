function dboundary1dgauss
    
    g1 = GaussDist(0,1);
    subplot(1,3,1);
    plot(g1,'plotArgs',{'LineWidth',2.5});
    xlabel('N(0,1)');
    subplot(1,3,2);
    g2 = GaussDist(1,1e6);
    plot(g2,'plotArgs',{'LineWidth',2.5});
    xlabel('N(1,1e6)');
    subplot(1,3,3);
    plot(g1,'plotArgs',{'LineWidth',2.5},'xrange',[-6,6]); hold on;
    plot(g2,'plotArgs',{'LineWidth',2.5},'xrange',[-6,6]);
    axis([-6,6,3.8e-4,4.2e-4]);
    set(gca,'XTick',[-3.7171,3.7171]);
    xlabel('R1');
    set(gcf,'Position',[265,535,751,287]);
    pdfcrop;
    
    
end