x_init = randn;

x_test = randn;

[a,b,c,fa,fb,fc] = bracket(@poly,x_init,x_test,poly(x_init),poly(x_test));

%% Plot
clf; hold on;
plot(min([a b c]):.01:max([a b c]),poly(min([a b c]):.01:max([a b c])))
plot([a b c],[fa fb fc],'*');
%%

[xmin,fmin]=golden(@poly,a,b,c,fb)
[xmin,fmin]=brent(@poly,a,b,c,fb)