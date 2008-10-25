function [G, labels, z] = loadGraph(name,nnodes)

% Load known graph interaction models
% For details see
% http://vlado.fmf.uni-lj.si/pub/networks/data/Ucinet/UciData.htm

labels={};
z=[];
G=[];

switch name
    case 'randgroup'
        c=round(nnodes/6);
        G_c=eye(c,c);
        for i=1:c
            for j=i+1:c
                u=rand(1);
                if u<1/(c)
                    G_c(i,j)=1;
                    G_c(j,i)=1;
                end
            end
        end
        for j=1:nnodes
            u=rand(1);
            z(j)=find(cumsum(1/c*ones(c,1))>u,1);
        end
        G=G_c(z,z);
        [z ind]=sort(z);
        G=G(ind,ind);
        figure
        hintonw(G)
%         G_c
%         z
%         pause
        
    case 'clusterMB'
        c=round(nnodes/5);
        mu=rand(2,c);
%         mu
        for j=1:nnodes
            u=rand(1);
            z(j)=find(cumsum(1/c*ones(c,1))>u,1);
            y(:,j)=mu(:,z(j))+.02*randn(2,1);
        end
        G=zeros(nnodes,nnodes);
        for j=1:nnodes
            for k=j+1:nnodes
%                 if z(j)==z(k)
%                     G(j,k)=1;
%                     G(k,j)=1;
%                 else
                    dist(j,k)=sqrt((y(1,j)-y(1,k))^2+(y(2,j)-y(2,k))^2);%/((1+100*(z(j)==z(k))));
                    prob=tpdf(dist(j,k)*sqrt(nnodes)/.4,4)/.4;
                    %                 prob=mvnpdf(y(:,j),y(8:,k),.02*eye(2));
                    u=rand(1);
                    if u<prob
                        G(j,k)=1;
                        G(k,j)=1;
%                     end
                end
            end
        end
        
%         figure
%         plot(y(1,:),y(2,:),'.')
%         for i=1:nnodes
%             for j=i+1:nnodes
%                 if G(i,j)==1
%                     line([y(1,i),y(1,j)],[y(2,i),y(2,j)],'linewidth',1)
%                 end
%             end
%         end
        [z ind]=sort(z);
        G=G(ind,ind);
%         pause
%         figure
%         hintonw(G)
%         pause
       
    case 'bipartite'
        k=1;
        c=3;
        for i=1:2*c:nnodes
            z(i:i+c-1)=k;
            z(i+c:i+2*c-1)=k+1;
            k=k+2;
        end
        z=z(1:nnodes);
    case 'pairAR1'
%         for i=1:nnodes/2
%             G(2*i-1:2*i+2,2*i-1:2*i+2)=1;
%         end
%         G=G(1:nnodes,1:nnodes);
%         G=diag(ones(1,nnodes-1),1)+diag(ones(1,nnodes-2),2)...%+diag(ones(1,nnodes-3),3)...
%             +diag(ones(1,nnodes-1),-1)+diag(ones(1,nnodes-2),-2);%+diag(ones(1,nnodes-3),-3);  
        for i=1:nnodes/2
            z(2*i-1)=i;
            z(2*i)=i;
        end
    case 'group1'
        z=[1 1 2 2 2 2 3 3 3 3 4 4];
        G=zeros(12);
        G(z==1,z==1)=1;
        G(z==1,z==2)=1;
        G(z==1,z==3)=1;
        G(z==2,z==1)=1;
        G(z==2,z==2)=1;
        G(z==3,z==1)=1;
        G(z==4,z==4)=1;
    case 'rnd'
        for i=1:nnodes
            for j=i+1:nnodes
                u=rand(1);
                if u>.95
                    G(i,j)=1;
                    G(j,i)=1;
                end
            end
        end
    case 'rndgroup'
%         d=15;
        d=nnodes/2;
        for i=1:d
            z(2*i-1)=i;
            z(2*i)=i;
        end
        G=zeros(2*d);        
        for i=1:d
            for j=i:d
                u=rand(1);
                if u>.9
                    G(z==i,z==j)=1;
                    G(z==j,z==i)=1;
                end
            end
        end
        
        case 'rndgroupFX'
        d=nnodes/3;
        for i=1:d
            z(3*i-2:3*i)=i;            
        end
        G=zeros(nnodes);     
        % Block effect
        for i=1:d
            for j=i:d
                u=rand(1);
                if u>.9
                    G(z==i,z==j)=1;
                    G(z==j,z==i)=1;
                end
            end
        end
        % Random FX
        for i=1:nnodes
            for j=i:nnodes
                u=rand(1);
                if u>.95
                    G(i,j)=1;
                    G(j,i)=1;
                end
            end
        end
    case 'bipartite2G'
        d=nnodes/2;
        for i=1:round(d/2)
            z(i)=1;
        end
        for i=round(d/2)+1:d
            z(i)=2;
        end
        z(d+1:d+round(d/2))=3:2+round(d/2);
        z(d+round(d/2)+1:nnodes)=2+round(d/2)+1:d+2;
        G(z==1,z==1)=1;
        G(z==2,z==2)=1;        
        G(z==1,d+1:d+round(d/2))=1;
        G(d+1:d+round(d/2),z==1)=1;
        G(z==2,d+round(d/2)+1:nnodes)=1;
        G(d+round(d/2)+1:nnodes,z==2)=1;        
    case 'zachary'
        G=[0 1 1 1 1 1 1 1 1 0 1 1 1 1 0 0 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 1 0 0
         1 0 1 1 0 0 0 1 0 0 0 0 0 1 0 0 0 1 0 1 0 1 0 0 0 0 0 0 0 0 1 0 0 0
         1 1 0 1 0 0 0 1 1 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 1 0
         1 1 1 0 0 0 0 1 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
         1 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
         1 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
         1 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
         1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
         1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 1
         0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
         1 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
         1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
         1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
         1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
         0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1
         0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1
         0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
         1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
         0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1
         1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
         0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1
         1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
         0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1
         0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 1 0 0 1 1
         0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 1 0 0
         0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 0 0
         0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1
         0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 1
         0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1
         0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 1 1
         0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1
         1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 1 0 0 0 1 1
         0 0 1 0 0 0 0 0 1 0 0 0 0 0 1 1 0 0 1 0 1 0 1 1 0 0 0 0 0 1 1 1 0 1
         0 0 0 0 0 0 0 0 1 1 0 0 0 1 1 1 0 0 1 1 1 0 1 1 0 0 1 1 1 1 1 1 1 0];
    case 'monk'
        labels={'ROMUL 10','BONAVEN 5','AMBROSE 9','BERTH 6','PETER 4','LOUIS 11',...
            'VICTOR 8', 'WINF 12', 'JOHN 1', 'GREG 2', 'HUGH 14', 'BONI 15', 'MARK 7',...
            'ALBERT 16','AMAND 13','BASIL 3','ELIAS 17','SIMP 18'};
        G=[0 1 1 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0
        0 0 2 0 3 3 0 0 0 1 0 0 0 0 0 0 0 0
        0 1 0 0 2 0 2 1 2 1 0 0 0 0 0 0 0 0
        0 1 3 0 4 2 0 0 0 1 0 0 0 0 0 0 0 0
        3 1 0 4 0 4 0 0 0 0 0 0 0 0 0 0 0 0
        0 3 2 0 2 0 2 0 0 0 1 0 0 1 0 0 0 0
        0 1 2 3 4 2 0 0 1 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 3 3 1 0 2 0 0 0 0 0
        0 1 0 0 0 0 1 4 0 1 2 0 1 0 0 1 1 0
        0 1 0 0 0 0 1 3 4 0 0 0 3 0 0 0 0 0
        0 0 0 0 0 0 0 3 4 3 0 4 0 1 0 0 0 0
        0 0 0 0 0 0 0 1 2 4 3 0 2 0 0 0 0 0
        0 0 0 0 0 0 0 3 0 4 0 2 0 4 0 0 0 0
        0 0 0 0 0 0 0 1 0 4 0 4 4 0 0 0 0 0
        0 4 0 0 0 3 0 0 0 0 0 0 3 0 0 0 0 1
        0 0 0 0 0 0 0 0 4 0 0 0 0 0 4 0 4 2
        0 0 0 0 0 0 0 0 0 3 0 0 0 0 1 3 0 3
        0 0 0 0 0 0 0 0 1 4 0 0 0 0 0 3 4 0];
        G=G>0;        
        z=[ones(1,7),2*ones(1,7),3*ones(1,4)];        
    case 'puremonk'
        % monk data with pure clusters
        [G labels z]=loadGraph('monk');
        G=zeros(size(G));
        G(z==1,z==1)=1;
        G(z==2,z==2)=1;
        G(z==3,z==3)=1;
    case 'puremonkFX'
                [G labels z]=loadGraph('puremonk');
                        for j=1:nnodes
            for k=j+1:nnodes
                u=rand(1);
                if u<.2
                    G(j,k)=1;
                    G(k,j)=1;
                end
            end
        end
    case 'wolf'
         G=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
         0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
         0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
         0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
         0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
         0 0 1 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0
         0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
         0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
         1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0
         0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
         1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0
         0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
         0 1 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0
         0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
         0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
         1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0
         0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
         0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
         1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
         0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
    case 'thurma'
        labels={'ANN','AMY','KATY','BILL','PETE',...
            'TINA','ANDY','LISA','PRESIDENT','MINNA',...
            'MARY','EMMA','ROSE','MIKE','PEG'};
        G=[0 1 1 0 1 1 0 1 1 0 1 0 1 0 0
         1 0 1 1 1 1 0 1 0 0 0 0 0 0 0
         1 1 0 0 1 1 0 1 0 0 0 0 0 0 0
         0 1 0 0 0 0 1 0 0 1 0 0 0 0 0
         1 1 1 0 0 1 1 1 1 0 0 1 0 0 0
         1 1 1 0 1 0 0 1 0 0 0 0 0 0 0
         0 0 0 1 1 0 0 0 0 1 0 0 0 0 0
         1 1 1 0 1 1 0 0 1 0 0 1 0 0 0
         1 0 0 0 1 0 0 1 0 0 0 1 0 0 0
         0 0 0 1 0 0 1 0 0 0 0 1 0 0 0
         1 0 0 0 0 0 0 0 0 0 0 1 0 0 0
         0 0 0 0 1 0 0 1 1 1 1 0 1 1 1
         1 0 0 0 0 0 0 0 0 0 0 1 0 0 0
         0 0 0 0 0 0 0 0 0 0 0 1 0 0 0
         0 0 0 0 0 0 0 0 0 0 0 1 0 0 0];
    case 'taro'
        G=[0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0
         1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0
         0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0
         0 0 1 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
         1 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0
         0 0 0 1 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
         0 0 0 1 0 1 0 1 0 0 0 0 0 0 0 1 0 0 1 0 0 0
         0 0 0 0 0 0 1 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0
         0 0 0 0 0 0 0 1 0 1 0 0 1 0 0 0 0 0 0 0 0 0
         0 0 0 0 0 0 0 1 1 0 1 0 0 0 0 0 0 0 0 0 0 0
         0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 1 0 1 1 0
         0 0 0 0 0 0 0 0 0 0 1 0 1 1 1 0 0 0 1 0 0 0
         0 0 0 0 0 0 0 0 1 0 0 1 0 1 0 0 0 0 0 0 0 0
         0 0 0 0 0 0 0 0 0 0 0 1 1 0 1 0 0 0 0 0 0 0
         0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 1 0 0 0 0 0 0
         0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0
         1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 1
         0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 1
         0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1
         0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0
         0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0
         0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0];
%     otherwise
%         error(['Unknown graph name: <' name '>']);
end
        