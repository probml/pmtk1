function er=NB_cv(X,Y,K)
% K-fold cross valdiation of a naive bayes classifier
% Based on code written by  Greg Shakhnarovich  MIT AI Lab  2002

% partition the data (do the two classes separately since they might have
% different number of examples in X)
i1=find(Y==1);
i2=find(Y==2);
X1=X(i1,:);
X2=X(i2,:);
nk1=floor(size(X1,1)/K);
nk2=floor(size(X2,1)/K);


for k=1:K
  if (K > 1)
    % get the (K-1)/K of the data and estimate the thetas on it
    x1=X1([1:(k-1)*nk1,k*nk1+1:size(X1,1)],:);
    x2=X2([1:(k-1)*nk2,k*nk2+1:+size(X2,1)],:);
    x=[x1;x2];
    y=[ones(size(x1,1),1);2*ones(size(x2,1),1)];
  else
    x = X;
    y = Y;
  end
  %theta = NBest(x,y);
  theta = NBtrain(x,y);

  % get the remainder for testing
  if (K > 1)
    xt1 = X1(((k-1)*nk1+1):min(k*nk1,size(X1,1)),:);
    xt2 = X2(((k-1)*nk2+1):min(k*nk2,size(X2,1)),:);
    xt = [xt1;xt2];
    yt=[ones(size(xt1,1),1);2*ones(size(xt2,1),1)];
  else
    xt = x;
    yt = y;
  end
  
  %yp=NBclas(xt,theta);
  yp = NBapply(xt, theta);
  er(k)=sum(yp~=yt)/length(yt);
end

% average over the K 'folds'
er = mean(er);
