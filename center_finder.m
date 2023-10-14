t = Tiff('10-13-23 array size calibration pic 2 lower power threshhold.tif');
image = read(t);
image = image(:,:,1);
[cent, cm] = FastPeakFind(image,60);
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

centerssort = sortrows(centers,1,'descend');
%% 

freq0 = [22;22;22;22;22;22;22;22;22;22;
    23;23;23;23;23;23;23;23;23;23;
    24;24;24;24;24;24;24;24;24;24;
    25;25;25;25;25;25;25;25;25;25;
    26;26;26;26;26;26;26;26;26;26;
    27;27;27;27;27;27;27;27;27;27];
freq1 = [23;23.5;24;24.5;25;25.5;26;26.5;27;27.5;
    23;23.5;24;24.5;25;25.5;26;26.5;27;27.5;
    23;23.5;24;24.5;25;25.5;26;26.5;27;27.5;
    23;23.5;24;24.5;25;25.5;26;26.5;27;27.5;
    23;23.5;24;24.5;25;25.5;26;26.5;27;27.5;
    23;23.5;24;24.5;25;25.5;26;26.5;27;27.5];
xdisp = (xpos(:) - xpos(35)) .* 4.8;
ydisp = (ypos(:) - ypos(35)) .* 4.8;
fit0 = polyfit(freq0, xdisp, 1);
xfit = fit0(1)*freq0 + fit0(2);
xfitin = fit0(1)/7;
fit1 = polyfit(freq1, ydisp, 1);
yfit = fit1(1)*freq1 + fit1(2);
yfitin = fit1(1)/7;
f1 = figure;
plot(freq0, xdisp,'*');
hold on;
plot(freq0, xfit, '--');
title('CH0 Displacement vs frequency');
xlabel('CH0 Output Frequency (MHz)');
ylabel('Displacement from Center (um)');
fitlabel = append(num2str((fit0(1))),' um/MHz ->', num2str((xfitin)), ' um/MHz inside');
legend('Spot center', fitlabel, 'Location', 'southeast');

f2 = figure;
plot(freq1, ydisp, '*');
hold on;
plot(freq1, yfit, '--');
title('CH1 Displacement vs Frequency');
xlabel('CH1 Output Frequency (MHz)');
ylabel('Displacement from Center (um)');
fitlabel = append(num2str((fit1(1))),' um/MHz ->', num2str((yfitin)), ' um/MHz inside');
legend('Spot center', fitlabel, 'Location', 'southeast');


fit3 = polyfit(freq1(11:20), xdisp(11:20), 1);
xfitoff = fit3(1)*freq1(11:20) + fit3(2);
xfitoffcham = fit3(1)/7;
fit4 = polyfit(freq0([1,11,21,31,41,51]), ydisp([1,11,21,31,41,51]), 1);
yfitoff = fit4(1)*freq0([1,11,21,31,41,51]) + fit4(2);
yfitoffcham = fit4(1)/7;

f3 = figure;
plot(freq1(11:20), xdisp(11:20),'*');
hold on;
plot(freq1(11:20), xfitoff, '--');
title('CH1 Displacement vs frequency Off Axis');
xlabel('CH1 Output Frequency (MHz)');
ylabel('Displacement from Center (um)');
fitlabel = append(num2str((fit3(1))),' um/MHz ->', num2str((xfitoffcham)), ' um/MHz inside');
legend('Spot center', fitlabel, 'Location', 'southeast');


f4 = figure;
plot(freq0([1,11,21,31,41,51]), ydisp([1,11,21,31,41,51]),'*');
hold on;
plot(freq0([1,11,21,31,41,51]), yfitoff, '--');
title('CH0 Displacement vs frequency Off Axis');
xlabel('CH0 Output Frequency (MHz)');
ylabel('Displacement from Center (um)');
fitlabel = append(num2str((fit4(1))),' um/MHz ->', num2str((yfitoffcham)), ' um/MHz inside');
legend('Spot center', fitlabel, 'Location', 'southeast');
