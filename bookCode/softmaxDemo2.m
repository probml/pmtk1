
figure(1);clf
nr = 1; nc = 4;
subplot(nr,nc,1);bar(softmax([3 0 1]/100)); title(sprintf('%s=100','T'))
subplot(nr,nc,2);bar(softmax([3 0 1]/1)); title(sprintf('%s=1','T'))
subplot(nr,nc,3);bar(softmax([3 0 1]/0.1)); title(sprintf('%s=0.1','T'))
subplot(nr,nc,4);bar(softmax([3 0 1]/0.01)); title(sprintf('%s=0.01','T'))

