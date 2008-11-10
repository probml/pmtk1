function gibbs2Ddemo
% Example of Gibbs sampling in a 2d Gaussian

      m = MvnDist;
      setSeed(0);
      m.Sigma = [1 -0.5; -0.5 1];
      m.mu = [1; 1];
      m = enterEvidence(m);
      marg1 = m.marginal(1);
      marg2 = m.marginal(2);
      m.stateInfEng = MvnMcmcInfer('method', 'gibbs', 'Nsamples', 500);   
      m = enterEvidence(m);
      X = m.sample(3);
      
%% Plotting Code      
      ax = @(parent,pos) axes('Parent',parent,'Position',pos);    
      fig1 = figure;
      mainAxf1 = ax(fig1,[0.13 0.11 0.775 0.815]);
      p1f1 = plot(m,'plotArgs',{'LineWidth',2});
      set(mainAxf1,'XTick',[],'YTick',[],'box','on','LineWidth',2);
      legend(p1f1,{'p(x)'},'FontSize',18,'Location','NorthEast');
      xlabel('Parent',mainAxf1,'x_1','FontSize',18); ylabel('Parent',mainAxf1,'x_2','FontSize',18);
      pdfcrop;
%%      
      fig2 = figure;  
      mainAxf2 = ax(fig2,[0.13,0.35,0.775,0.57]); hold on;
      margAxf2 = ax(fig2,[0.13,0.02,0.775,0.25]);
      p1f2 = copyobj(p1f1,mainAxf2);
      p2f2 = plot(X(1,1),X(1,2),'.k','MarkerSize',30,'Parent',mainAxf2);
      p3f2 = plot(X(2,1),X(1,2),'ok','MarkerSize',8,'MarkerFaceColor',0.9*[1 1 1],'LineWidth',2.5,'Parent',mainAxf2);
      annotation(fig2   , 'textarrow' ,[0.6798 0.7685],[0.3961 0.3971],...
        'TextEdgeColor' , 'none'      ,...
        'TextLineWidth' , 2           ,...
        'FontSize'      , 14          ,...
        'String'        , {'x^{(t)}'} ,...
        'LineWidth'     , 2           ,...
        'Color'         , [1 0 0]     );

      set(mainAxf2,'XTick',[],'YTick',[],'box','on','LineWidth',2);
      legend(p1f2,{'p(x)'},'FontSize',18,'Location','NorthEast');
      p3 = plot(marg1,'plotArgs',{'Parent',margAxf2,'LineWidth',3,'Color','r'});
      set([mainAxf2,margAxf2],'XTick',[],'YTick',[],'box','on','LineWidth',2);
      axis([-2 4 0 0.5]);
      legend(p3,{'p(x_1 | x_2^{(t)})'},'FontSize',14,'Location','NorthEast');
      pdfcrop;
%%
      fig3 = figure; 
      mainAxf3 = ax(fig3,[0.3,0.11,0.65,0.815]); hold on;
      margAxf3 = ax(fig3,[0.05 0.11 0.22 0.815]);hold on;
      p1f3 = plot(marg2,'plotArgs',{'Parent',margAxf3,'LineWidth',3,'Color','b'});
      view([90 90]);
      p2f3 = copyobj(p1f1,mainAxf3);
      p3f3 = copyobj(p2f2,mainAxf3);
      p4f3 = copyobj(p3f2,mainAxf3);
      p5f3 = plot(X(2,1),X(2,2),'.k','MarkerSize',30,'Parent',mainAxf3);
      axis([-2 4 0 0.5]);
      annotation(fig3,'arrow',[0.9 0.9],[0.2242 0.4286],'LineWidth',2,'Color',[0 0 1]);
      legend(p1f3,{'p(x_2 | x_1)'},'FontSize',14,'Location','NorthOutside');
      legend(p2f3,{'p(x)'},'FontSize',18,'Location','NorthEast');
      set([mainAxf3,margAxf3],'XTick',[],'YTick',[],'box','on','LineWidth',2);
      pdfcrop;
%%      
      fig4 = figure;
      mainAxf4 = ax(fig4,[0.13 0.11 0.775 0.815]); hold on;
      p1f4 = copyobj(p1f1,mainAxf4);
      p2f4 = copyobj(p3f3,mainAxf4);
      p3f4 = copyobj(p4f3,mainAxf4);
      p4f4 = copyobj(p5f3,mainAxf4);
      p4f5 = plot(X(3,1),X(2,2),'ok','MarkerSize',8,'MarkerFaceColor',0.9*[1 1 1],'LineWidth',2.5,'Parent',mainAxf4);
      p5f5 = plot(X(3,1),X(3,2),'.k','MarkerSize',30,'Parent',mainAxf4);
      set(mainAxf4,'XTick',[],'YTick',[],'box','on','LineWidth',2);
      legend(p1f4,{'p(x)'},'FontSize',18,'Location','NorthEast');
      xlabel('Parent',mainAxf4,'x_1','FontSize',18); ylabel('Parent',mainAxf4,'x_2','FontSize',18);
      
      
    arrowSpec = {'HeadLength',5,'HeadWidth',6,'HeadStyle','plain','LineWidth',2};    
    annotation(fig4,'arrow',[0.6820 0.7500],[0.2425 0.2425],'Color','r',arrowSpec{:});
    annotation(fig4,'arrow',[0.7075 0.7075],[0.4000 0.3250],'Color','b',arrowSpec{:});
    annotation(fig4,'arrow',[0.7520 0.7210],[0.4190 0.4190],'Color','r',arrowSpec{:});
    annotation(fig4,'arrow',[0.7650 0.7650],[0.2580 0.4000],'Color','b',arrowSpec{:});

    textSpec = {'FontSize',16,'FontWeight','bold','LineStyle','none'};
    
    annotation(fig4,'textbox',[0.65 0.159 0.0604 0.09443],...
    'String',{'x^{(t)}'},textSpec{:});

    annotation(fig4,'textbox',[0.735 0.419 0.07942 0.09443],...
    'String',{'x^{(t+1)}'},textSpec{:});
   
    annotation(fig4,'textbox',[0.60 0.2578 0.07942 0.09443],...
    'String',{'x^{(t+2)}'},textSpec{:});
     
    pdfcrop;
      
      
    end