t = Tiff('20230524 ch0 spread 21-29MHz mask.tif');
image = read(t);
image = image(:,:,1);
[cent cm] = FastPeakFind(image,60);
mesh(cm);

%% 


xpos= [];
ypos= [];


for n =1 : length(cent)
    if mod(n,2) == 1
        xpos = [xpos cent(n)];
    end
    if mod(n,2) == 0
        ypos = [ypos cent(n)];
    end
end

xpos = xpos.';
ypos = ypos.';
centers = [xpos ypos];

centerssort = sortrows(centers,1,'descend')
%% 

freq = [-4;-3;-2;-1;0;1;2;3;4];
xdisp = (centerssort(:,1) - centerssort(4,1)) .* 5.2;
ydisp = (centerssort(:,2) - centerssort(4,2)) .* 5.2;
fit0 = polyfit(freq, xdisp, 1);
xfit = fit0(1)*freq + fit0(2);
xfitin = fit0(1)/7;
fit1 = polyfit(freq, ydisp, 1);
yfit = fit1(1)*freq + fit1(2);
yfitin = fit1(1)/7;
f1 = figure;
plot(freq, xdisp,'*');
hold on;
plot(freq, xfit, '--');
title('CH0 Displacement vs frequency On Axis');
xlabel('CH0 Output Frequency (MHz)');
ylabel('Displacement from Center (um)');
fitlabel = append(num2str((fit0(1))),' um/MHz ->', num2str((xfitin)), ' um/MHz inside');
legend('Spot center', fitlabel, 'Location', 'southwest');

f2 = figure;
plot(freq, ydisp, '*');
hold on;
plot(freq, yfit, '--');
title('CH0 Displacement vs frequency Off Axis');
xlabel('CH0 Output Frequency (MHz)');
ylabel('Displacement from Center (um)');
fitlabel = append(num2str((fit1(1))),' um/MHz ->', num2str((yfitin)), ' um/MHz inside');
legend('Spot center', fitlabel, 'Location', 'southwest');

