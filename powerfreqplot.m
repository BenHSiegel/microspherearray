


%2-14-22 ch1 sweep data (broken channel)
startsweepdelay = 8.373;

scantime = table2array(CH1frequencysweep(:,1));
scantime = scantime - scantime(1);

frequencies = table2array(CH1frequencysweep(:,3));

powertime = table2array(ch1powerspectrum(:,1));
powertime = powertime - startsweepdelay;

power = table2array(ch1powerspectrum(:,2));

[frequencies214, avgpower214] = freqtopower(scantime, powertime, power, frequencies);

inputpower = 1.138;
normpower214 = avgpower214/inputpower;
figure();
plot(frequencies214, normpower214);
xlabel('Frequency Offset from Center (MHz)');
ylabel('Efficiency (%)');
title('Power Spectrum of CH1 (Broken Channel)');
hold on;


%% 
%2-16-22 ch0sweep data
startsweepdelay = 3.966;

scantime = table2array(Ch0freqsweep(:,1));
scantime = scantime - scantime(1);

frequencies = table2array(Ch0freqsweep(:,3));

powertime = table2array(Ch0power(:,1));
powertime = powertime - startsweepdelay;

power = table2array(Ch0power(:,2));

[frequencies, avgpower] = freqtopower(scantime, powertime, power, frequencies);

figure();
plot(frequencies, avgpower);
xlabel('Frequency Offset from Center (MHz)');
ylabel('Power [W]');
title('2-16-22 Power Spectrum of CH0');

inputpower = 1.216;

%% 

%2-21-22 ch0sweep data
startsweepdelay = 5.185;

scantime = table2array(CH1freqsweep2_21(:,1));
scantime = scantime - scantime(1);

frequencies = table2array(CH1freqsweep2_21(:,3));

powertime = table2array(CH1power2_21(:,1));
powertime = powertime - startsweepdelay;

power = table2array(CH1power2_21(:,2));

[frequencies221, avgpower221] = freqtopower(scantime, powertime, power, frequencies);

inputpower = 1.280;
normpower221 = avgpower221/inputpower;

figure();
plot(frequencies214, avgpower214);
hold on;
plot(frequencies221, avgpower221);
xlabel('Frequency Offset from Center (MHz)');
ylabel('Power [W]');
title('Power Spectrum of CH1 (Broken Channel)');
legend('2-14 data', '2-21 data')
%% 
%4-14-22 Power sweep data for CH0 averaging over 3 runs

startsweepdelay = 4.315;

ch0frequencyslowscan1 = importdata('ch0frequency_slowscan1.txt');

scantime = cell2mat(cellfun(@str2num,ch0frequencyslowscan1.textdata(:,1),'un',0));
scantime = scantime - scantime(1);

frequencies = ch0frequencyslowscan1.data;

ch0powerslowscan1 = importdata('ch0power_slowscan1.txt').data;

powertime = ch0powerslowscan1(:,1);
powertime = powertime - startsweepdelay;

power = ch0powerslowscan1(:,2);

[frequenciesch0scan1, avgpowerch0scan1] = freqtopower(scantime, powertime, power, frequencies);

inputpower = 0.107;
normpowerch0scan1 = avgpowerch0scan1/inputpower;



startsweepdelay = 15.703;

ch0frequencyslowscan2 = importdata('ch0frequency_slowscan2.txt');

scantime = cell2mat(cellfun(@str2num,ch0frequencyslowscan2.textdata(:,1),'un',0));
scantime = scantime - scantime(1);

frequencies = ch0frequencyslowscan2.data;

ch0powerslowscan2 = importdata('ch0power_slowscan2.txt').data;

powertime = ch0powerslowscan2(:,1);
powertime = powertime - startsweepdelay;

power = ch0powerslowscan2(:,2);

[frequenciesch0scan2, avgpowerch0scan2] = freqtopower(scantime, powertime, power, frequencies);

inputpower = 0.107;
normpowerch0scan2 = avgpowerch0scan2/inputpower;




startsweepdelay = 7.777;

ch0frequencyslowscan3 = importdata('ch0frequency_slowscan3.txt');

scantime = cell2mat(cellfun(@str2num,ch0frequencyslowscan3.textdata(:,1),'un',0));
scantime = scantime - scantime(1);

frequencies = ch0frequencyslowscan3.data;

ch0powerslowscan3 = importdata('ch0power_slowscan3.txt').data;

powertime = ch0powerslowscan3(:,1);
powertime = powertime - startsweepdelay;

power = ch0powerslowscan3(:,2);

[frequenciesch0scan3, avgpowerch0scan3] = freqtopower(scantime, powertime, power, frequencies);

inputpower = 0.115;
normpowerch0scan3 = avgpowerch0scan3/inputpower;



normpowerch0 = (normpowerch0scan1 + normpowerch0scan2 + normpowerch0scan3)/3;


figure();
hold on;
plot(frequenciesch0scan1, 100*normpowerch0scan1);
plot(frequenciesch0scan2, 100*normpowerch0scan2);
plot(frequenciesch0scan3, 100*normpowerch0scan3);
plot(frequenciesch0scan1, 100*normpowerch0);
xlabel('Frequency Offset from Center (MHz)');
ylabel('Efficiency');
title('Power Spectrum of CH0');
legend('Scan 1', 'Scan 2', 'Scan 3', 'Average');

ch0freqvpower = cat(2, frequenciesch0scan1, normpowerch0);
%csvwrite('4-14ch0freqvpower.csv', ch0freqvpower);

%% 

%7-20-22 Power sweep data for CH1

startsweepdelay = 1.301;

ch1freqdata = importdata('CH1sweep-5to5.txt');

scantime = cell2mat(cellfun(@str2num,ch1freqdata.textdata(:,1),'un',0));
scantime = scantime - scantime(1);

frequencies = ch1freqdata.data;

ch1powerdata = importdata('CH1power-5to5.txt').data;

powertime = ch1powerdata(:,1);
powertime = powertime - startsweepdelay;

power = ch1powerdata(:,2);

[frequenciesch1, avgpowerch1] = freqtopower(scantime, powertime, power, frequencies);

inputpower = 1.18;
efficiencych1 = 100*avgpowerch1/inputpower;
normpowerch1 = avgpowerch1/max(avgpowerch1);


figure();
hold on;
plot(frequenciesch1, efficiencych1);
plot(frequenciesch1, 100*normpowerch1);
xlabel('Frequency Offset from Center (MHz)');
ylabel('Efficiency');
title('Power Spectrum of CH1');
ch1freqvpower = cat(2, frequenciesch1, normpowerch1);
%csvwrite('7-20ch1freqvpower.csv', ch1freqvpower);

%% 
%7-20-22 Power sweep data for CH0

startsweepdelay = 1.155;

ch0freqdata = importdata('CH0sweep-5to5.txt');

scantime = cell2mat(cellfun(@str2num,ch0freqdata.textdata(:,1),'un',0));
scantime = scantime - scantime(1);

frequencies = ch0freqdata.data;

ch0powerdata = importdata('CH0power-5to5.txt').data;

powertime = ch0powerdata(:,1);
powertime = powertime - startsweepdelay;

power = ch0powerdata(:,2);

[frequenciesch0, avgpowerch0] = freqtopower(scantime, powertime, power, frequencies);

inputpower = 1.086;
efficiencych0 = 100*avgpowerch0/inputpower;
normpowerch0 = avgpowerch0/max(avgpowerch0);


figure();
hold on;
plot(frequenciesch0, efficiencych0);
plot(frequenciesch1, 100*normpowerch0);


xlabel('Frequency Offset from Center (MHz)');
ylabel('Efficiency');
title('Power Spectrum of CH1');
ch0freqvpower = cat(2, frequenciesch0, normpowerch0);
%csvwrite('7-20ch0freqvpower.csv', ch0freqvpower);

%% 
%8-4-22 Power Sweep for Channel 0
startsweepdelay = 2.685;

ch0freqdata = importdata('ch0_nonorm_-4to4MHz.txt');

scantime = cell2mat(cellfun(@str2num,ch0freqdata.textdata(:,1),'un',0));
scantime = scantime - scantime(1);

frequencies = ch0freqdata.data;

ch0powerdata = importdata('ch0power_nonorm_-4to4MHz.txt').data;

powertime = ch0powerdata(:,1);
powertime = powertime - startsweepdelay;

power = ch0powerdata(:,2);

[frequenciesch0, avgpowerch0] = freqtopower(scantime, powertime, power, frequencies);

inputpower = 0.990;
efficiencych0 = 100*avgpowerch0/inputpower;
normpowerch0 = avgpowerch0/max(avgpowerch0);


figure();
hold on;
plot(frequenciesch0, efficiencych0);
plot(frequenciesch0, 100*normpowerch0);


xlabel('Frequency Offset from Center (MHz)');
ylabel('Normalized Power (% of Max Transmitted)');
title('Power Spectrum of CH0');
ch0freqvpower = cat(2, frequenciesch0, normpowerch0);
%csvwrite('8-4ch0freqvpower.csv', ch0freqvpower);

%% 
%% 
%8-4-22 Power Sweep for Channel 1
startsweepdelay = 2.685;

ch1freqdata = importdata('ch1_nonorm_-4to4MHz.txt');

scantime = cell2mat(cellfun(@str2num,ch1freqdata.textdata(:,1),'un',0));
scantime = scantime - scantime(1);

frequencies = ch1freqdata.data;

ch1powerdata = importdata('ch1power_nonorm_-4to4MHz.txt').data;

powertime = ch1powerdata(:,1);
powertime = powertime - startsweepdelay;

power = ch1powerdata(:,2);

[frequenciesch1, avgpowerch1] = freqtopower(scantime, powertime, power, frequencies);

inputpower = 0.990;
efficiencych1 = 100*avgpowerch1/inputpower;
normpowerch1 = avgpowerch1/max(avgpowerch1);


figure();
hold on;
%plot(frequenciesch1, efficiencych1);
plot(frequenciesch1, 100*normpowerch1);


xlabel('Frequency Offset from Center (MHz)');
ylabel('Normalized Power (% of Max Transmitted)');
title('Power Spectrum of CH1');
ch1freqvpower = cat(2, frequenciesch1, normpowerch1);
%csvwrite('8-4ch1freqvpower.csv', ch0freqvpower);


%% 

%8-5-22 Gain power check ch0
startsweepdelay = 2.580;

gaindata = importdata('ch0check2.txt');

scantime = cell2mat(cellfun(@str2num,gaindata.textdata(:,1),'un',0));
scantime = scantime - scantime(1);

gainvalues = gaindata.data(:,2);

powerdata = importdata('ch0powercheck2.txt').data;

powertime = powerdata(:,1);
powertime = powertime - startsweepdelay;

power = powerdata(:,2);

[gains, avgpower] = freqtopower(scantime, powertime, power, gainvalues);
normpower = 100*avgpower/max(avgpower);

hold on;
plot(gains,normpower, '.')
plot(gains,gains)
%ch0gainvpower = cat(2, gains, normpower);
%csvwrite('8-5ch0gainvpower.csv', ch0gainvpower);
%% 

%8-8-22 Gain power check ch1
startsweepdelay = 3.111;

gaindata = importdata('ch1gaincheck3.txt');

scantime = cell2mat(cellfun(@str2num,gaindata.textdata(:,1),'un',0));
scantime = scantime - scantime(1);

gainvalues = gaindata.data(:,2);

powerdata = importdata('ch1gaincheckpower3.txt').data;

powertime = powerdata(:,1);
powertime = powertime - startsweepdelay;

power = powerdata(:,2);

[gains, avgpower] = freqtopower(scantime, powertime, power, gainvalues);
normpower = 100*avgpower/max(avgpower);

hold on;
plot(gains,normpower, '.')
plot(gains,gains)
%ch1gainvpower = cat(2, gains, normpower);
%csvwrite('8-8ch1gainvpower2.csv', ch1gainvpower);
%% 

%8-8-22 CH0 gain correction data
%plotgainpower(1.308, 'ch0gaincorrection.txt', 'ch0gaincorrectionpower.txt', true, '8-8ch0gainvpower.csv');

%8-8-22 CH0 gain correction check
plotgainpower(2.118, 'ch0gaincheck.txt', 'ch0gaincheckpower.txt', false, ' ');

%8-8-22 CH0 gain check at different frequency
plotgainpower(3.336, 'ch0gaincheck-0.5MHz.txt', 'ch0gaincheckpower-0.5MHz.txt', false, ' ');

%8-8-22 CH0 gain check at a third frequency
plotgainpower(2.571, 'ch0gaincheck1MHz.txt', 'ch0gaincheckpower1MHz.txt', false, ' ');

%% 

%9-5-23 CH0 frequency to power data with new filter boards
%plotpowerspectrum(4.663, 'ch0freqscan15-35.txt', 'CH0 scan 15-35MHz.txt', 1.17, false, '9-5ch0freqvpower.csv');

%9-5-23 CH1 frequency to power data with new filter boards
%plotpowerspectrum(7.404, 'ch1freqscan15-35.txt', 'CH1 scan 15-35MHz.txt', 1.2, false, '9-5ch1freqvpower.csv');

%9-5-23 CH0 frequency to power data with new filter boards normalized to
%0.9 of its maximum power
%plotpowerspectrum(3.436, 'ch0freqscannormcheck.txt', 'CH0 scan normcheck.txt', 1.185, false, ' ');

%9-5-23 CH1 frequency to power data with new filter boards normalized to
%0.85 of its maximum power
plotpowerspectrum(4.362, 'ch1freqscannormcheck.txt', 'CH1 scan normcheck.txt', 1.185, false, ' ');
%% 

function plotgainpower(startsweepdelay, gaindatapath, powerdatapath, writetofile, writepath)


gaindata = importdata(gaindatapath);

scantime = cell2mat(cellfun(@str2num,gaindata.textdata(:,1),'un',0));
scantime = scantime - scantime(1);

gainvalues = gaindata.data(:,2);

powerdata = importdata(powerdatapath).data;

powertime = powerdata(:,1);
powertime = powertime - startsweepdelay;

power = powerdata(:,2);

[gains, avgpower] = freqtopower(scantime, powertime, power, gainvalues);
normpower = 100*avgpower/max(avgpower);

hold on;
plot(gains,normpower, '.')
plot(gains,gains)


if writetofile == true
    gainvpower = cat(2, gains, normpower);
    csvwrite(writepath, gainvpower);
end

end
%% 


function plotpowerspectrum(startsweepdelay, freqdatapath, powerdatapath, inputpower, writetofile, writepath)


freqdata = importdata(freqdatapath);

scantime = cell2mat(cellfun(@str2num,freqdata.textdata(:,1),'un',0));
scantime = scantime - scantime(1);

frequencies = freqdata.data;

powerdata = importdata(powerdatapath).data;

powertime = powerdata(:,1);
powertime = powertime - startsweepdelay;

power = powerdata(:,2);

[frequencies, avgpower] = freqtopower(scantime, powertime, power, frequencies);


efficiency = 100*avgpower/inputpower;
normpower = avgpower/max(avgpower);


figure();
plot(frequencies, efficiency);
xlabel('FPGA Generated Frequency (MHz)');
ylabel('Efficiency of AOD (% of input power)');
title('Efficiency Spectrum');

figure();
plot(frequencies, 100*normpower);
xlabel('FPGA Generated Frequency (MHz)');
ylabel('Normalized Power (% of Max Transmitted)');
title('Performance Spectrum');


freqvpower = cat(2, frequencies, normpower);
if writetofile == true
    csvwrite(writepath, freqvpower);
end

end


%% 


function [ frequencies, avgpower] = freqtopower(scantime, powertime, power, frequencies)

avgpower = zeros(length(scantime),1);
summation = zeros(length(scantime),1);
count = zeros(length(scantime),1);
for j=1:length(scantime)
    avg = 0;
    if j+1 > length(scantime)
        
    else
        for i=1:length(powertime)
          if  scantime(j) < powertime(i) && powertime(i) < scantime(j+1)
              summation(j,1) = summation(j,1) + power(i);
              count(j,1) = count(j,1) + 1;
          end
        end
    avgpower(j,1) = summation(j,1)/count(j,1);
    end
end

frequencies = frequencies(1:length(frequencies)-1,1);
avgpower = avgpower(1:length(avgpower)-1,1);
end



