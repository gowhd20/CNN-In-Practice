load('curve.mat', 'x', 'x2', 'y', 'y2');

eq = 'a*exp(-b.*x) + c*exp(-((x-d)/e).^2)';
startPoints = [3 0.1 1 300 50]; % Initial estimate
fitobject = fit(x, y, eq, 'Start', startPoints); % finds the parameters
figure,plot(fitobject,x,y);


initialPoints = [6 0 4 190 50];
fitobject2 = fit(x2, y2, eq, 'start', initialPoints); % trained parameters
figure,plot(fitobject2,x2,y2);




