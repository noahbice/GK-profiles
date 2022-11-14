close all; clear; clc;
visPlot = [1, 0, 0];    % [full plot, pointLine, save figures]

%% import data from Mod or Raw PCElectrometer file
% K = 3; % =1 for x only, =3 for xyz
% [id, Pos, Rdg] = importMod (dataFile, visPlot); % import data, check using figure, set initial values, 

% dataFileRaw = Raw PCElectrometer file
% dataFile =    '4mmX81ptsAir.xlsx'; N = 81; I0 = 4000; I1 = 1.3;   [id, Pos, Rdg] = importRaw (dataFile, N, I0, I1, visPlot); 
% dataFile =    '4mmX81ptsAir.xlsx'; N = 81; I0 = 2.1; I1 = 1.3;   [id, Pos, Rdg] = importRaw (dataFile, N, I0, I1, visPlot); 
%  dataFile =    '4mmY81pts.xlsx'; N = 81; I0 = 2.1; I1 = 1.3;  
% dataFile =    '4mmZ81pts.xlsx'; N = 81; I0 = 2.1; I1 = 1.3; 
% dataFile =    '8mmX81pts.xlsx'; N = 81; I0 = 2.1; I1 = 1.3; %I1p 10%  
% dataFile =    '8mmY81pts.xlsx'; N = 81; I0 = 2.1; I1 = 1.3; 
% dataFile =    '8mmZ81pts.xlsx'; N = 81; I0 = 2.1; I1 = 1.3;
dataFile =    '16mmX109pts.xlsx'; N = 109; I0 = 5; I1 = 1.5;   [id, Pos, Rdg] = importRaw (dataFile, N, I0, I1, visPlot);  
%  dataFile =    '16mmY109pts.xlsx'; N = 109; I0 = 2; I1 = 1.5;   [id, Pos, Rdg] = importRaw (dataFile, N, I0, I1, visPlot);  
% dataFile =    '16mmX81pts.xlsx'; N = 81; I0 = 2; I1 = 1.5;   [id, Pos, Rdg] = importRaw (dataFile, N, I0, I1, visPlot);  
% dataFile1 =    '16mmX20pts_tails.xlsx'; N = 20; I0 = 0.9; I1 = 0.3; [id1, Pos1, Rdg1] = importRaw1 (dataFile1, N, I0, I1, visPlot); 
% dataFile1 =    '16mmX11pts_40-60.xlsx'; N = 10; I0 = 0.9; I1 = 0.3; [id1, Pos1, Rdg1] = importRaw1 (dataFile1, N, I0, I1, visPlot); 
% dataFile1 =    '16mmX81pts.xlsx'; N = 101; I0 = 2.1; I1 = 1.3; 
% dataFile =    '16mmY81pts.xlsx'; N = 81; I0 = 3; I1 = 1.5;     [id, Pos, Rdg] = importRaw (dataFile, N, I0, I1, visPlot);  
% dataFile =    '16mmZ81pts.xlsx'; N = 81; I0 = 3; I1 = 1.5;  [id, Pos, Rdg] = importRaw (dataFile, N, I0, I1, visPlot); % dataFile =    '8mm_z_41pts.xlsx'; N = 41; I0 = 0.5; I1 = 1.0;  
% dataFile =    '16mm_x_40pts_missing83mm.xlsx'; N = 40; I0 = 1; I1 = 1.0; [id, Pos, Rdg] = importRaw (dataFile, N, I0, I1, visPlot);   % missing 83mm
K = 1; % =1 for x only, =3 for xyz
% [id, Pos, Rdg] = importRaw (dataFile, N, I0, I1, visPlot);


%% interperature: 
% [Pos_, FWHM_, R2_] = LinearInter(id, Pos, Rdg, K, visPlot); 

%% Gauss: call/fit, output
% [Pos_, FWHM_, R2_] = Gauss(id, Pos, Rdg, K, visPlot); 
% [Final] = output_Gauss(dataFile, t4mmIn_, t4mmOut_, rsquareIn_, rsquareOut_, visPlot);

%% Sigmoid: call/fit, output
% [tBetweenShots_, shutterIn_, shutterOut_, tDwell_, tEffect_, tError_, tErrPct_, rsquareIn_, rsquareOut_, heightIn_, heightOut_] = start_Sigmoid (id, Time, Current, cellAll, visPlot);
% [Final] = output_Sigmoid_all (dataFile, tBetweenShots_, shutterIn_, shutterOut_, tDwell_, tEffect_, tError_, tErrPct_, rsquareIn_, rsquareOut_, heightIn_, heightOut_, visPlot);



%% import data, and find initial guess for fitting functions
function [id, Pos, Rdg] = importRaw(dataFile, N, I0, I1, visPlot)
[data,txt,raw] = xlsread(dataFile);

logDat = string(txt(5,1));
fmt = '%*s %*s %s %s %s';  %  %*s is to ignore    https://www.mathworks.com/help/matlab/ref/textscan.html#inputarg_formatSpec
dat = textscan(logDat, fmt);
logDate = string(dat{1});
logTime = string(dat{2});
logAmPm = string(dat{3});
logDateTime = logDate + " " + logTime + " " + logAmPm;

index =(1:size(data(:,1)));
time = data(:,1);  
currentOriginal = data(:,3);
current = medfilt1(currentOriginal);
indexCenter = current > 4.99;
timeCenter = time (indexCenter);
currentCenter = current (indexCenter);
indexTails = current < 5.00;
timeTails = time (indexTails);
currentTails = current (indexTails);
ISize = length(current); 


IBKGD = mean(current(1:10)); % BKGD = background
IMax = max(current);
IHalf = IMax/2;     % for find index

if visPlot(1) == 1; figure(99); plot(time,current,'-r*'); grid on; end; % plot1=plot(x*2,y); % %plot2=plot(x*2,q,'*')

% store values in matrix in a loop  https://www.mathworks.com/matlabcentral/answers/120444-storing-values-from-nested-for-loop-array-only-saves-last-run-of-results
Ibg = abs(mean(current(1:10))); % current background

j = 0;
k = 0; % index of the shots

for i = 1:(ISize-5) % ISize/3-5+70
    A = current(i);
    B = abs(current(i + 1) - current(i)); 
    C = abs(current(i + 2) - current(i + 1));
    D = abs(current(i + 3) - current(i + 2));
    % E = abs(current(i + 4) - current(i + 3));
    
    if N == 81; Pos = (80:0.5:120)';
    elseif N == 109; Pos = [(40:5:55), (60:2:78), (80:0.5:119.5), (120:2:138), (140:5:160)];
    end
        % 40 points for 16mm, missing 83mm;
    
    if A > I0 && (B < I1 && C < I1 && D < I1) && (i-j)>5   % 1) > I0; 2) three pts together; 3) points not too close;
        k = k + 1;  % kth of the 81 pts;
        j = i; % index for the first point of 0.06min/0.5sec = 7.2 points of the kth shot
        p1 = Pos(k);
        time1 = time(i+3);
        current1 = current(i);
        c = mean([current(i+2), current(i+3), current(i+4)]); % middle three currents out of 7
        if visPlot(1)==1; hold on; plot(time1, c, '-bo'); hold off; end;
        k_(k) = k;
        j_(k) = j;  % value in time of kth pts of 81;
        p1_(k) = p1;
        c_(k) = current1;
        shot_(k,:) = [p1, current1];    % 81, the jth 0.5 sec, 12*0.5=6 sec apart;
    end
end
[IMax, kMax] = max(c_);
shot_(kMax-1,:);
shot_(kMax,:);
shot_(kMax+1,:);


id = ones(N,1);
Rdg = shot_(:,2);
% 
% fileID = fopen('16mmX.txt','w');
% fprintf(fileID, '%6f %6f.3 \n', Pos, Rdg);
writematrix(shot_);
end
function [id1, Pos1, Rdg1] = importRaw1(dataFile1, N, I0, I1, visPlot)
[data,txt,raw] = xlsread(dataFile1);

logDat = string(txt(5,1));
fmt = '%*s %*s %s %s %s';  %  %*s is to ignore    https://www.mathworks.com/help/matlab/ref/textscan.html#inputarg_formatSpec
dat = textscan(logDat, fmt);
logDate = string(dat{1});
logTime = string(dat{2});
logAmPm = string(dat{3});
logDateTime = logDate + " " + logTime + " " + logAmPm;

index =(1:size(data(:,1)));
time = data(:,1);  
current = data(:,3);  
ISize = length(current); 

IBKGD = mean(current(1:10)); % BKGD = background
IMax = max(current);
IHalf = IMax/2;     % for find index

if visPlot(1) == 1; figure(98); plot(time,current,'-r*'); grid on; end; % plot1=plot(x*2,y); % %plot2=plot(x*2,q,'*')

% store values in matrix in a loop  https://www.mathworks.com/matlabcentral/answers/120444-storing-values-from-nested-for-loop-array-only-saves-last-run-of-results
Ibg = abs(mean(current(1:10))); % current background

j = 0;
k = 0; % index of the shots

for i = 1:(ISize-5)
    A = current(i);
    B = abs(current(i + 1) - current(i)); 
    C = abs(current(i + 2) - current(i + 1));
    D = abs(current(i + 3) - current(i + 2));
    % E = abs(current(i + 4) - current(i + 3));
    if A > I0 && (B < I1 && C < I1 && D < I1) && (i-j)>5   % 1) > I0; 2) three pts together; 3) points not too close;
        k = k + 1;  % kth of the 81 pts;
        j = i; % index for the first point of 0.06min/0.5sec = 7.2 points of the kth shot
        time1 = time(i);
        current1 = current(i);
%         c = mean([current(i+2), current(i+3), current(i+4)]); % middle three currents out of 7
        if visPlot(1)==1; hold on; plot(time1, current1, '-bo'); hold off; end;
        k_(k) = k;
        j_(k) = j;  % value in time of kth pts of 81;
        c_(k) = current1;
        shot_1(k,:) = [k, time1, current1];    % 81, the jth 0.5 sec, 12*0.5=6 sec apart;
    end
end
[IMax, kMax] = max(c_);
shot_1(kMax-1,:)
shot_1(kMax,:)
shot_1(kMax+1,:)



id1 = ones(N,1);
Pos1 = (80:0.5:120)'; % 40 points for 16mm, missing 83mm;
Rdg1 = shot_1(:,2);
end


%% LinearInter
function [Pos_, FWHM_, R2_] = LinearInter(id, Pos, Rdg, K, visPlot); 

for i = 1:K
iPos = Pos (id == i);
iRdg = Rdg (id == i);
visPlot(2) =1; if visPlot(2) == 1  figure(i*100); plot (iPos, iRdg, 'b-*'); hold on; end; %xlabel ('Position (mm)'); ylabel (' Charge(relative) '); title ('Source - Target(i)');   grid on; set(gcf, 'units','mm', 'position', [i+3, i+1, 30, 15]); ylim([-10 60]); hold on; end

% Max func
[iRdgMax, idMax] = max(iRdg)
iPosMax = iPos(idMax);
Center = iPosMax

% Half Max https://www.mathworks.com/matlabcentral/answers/310113-how-to-find-out-full-width-at-half-maximum-from-this-graph
halfMax = iRdgMax / 2;
halfMaxIndex1 = find(iRdg >= halfMax, 1, 'first')
halfMaxIndex2 = find(iRdg >= halfMax, 1, 'last')
FWHM = iPos(halfMaxIndex2) - iPos(halfMaxIndex1)    %FWHM LGP: 4mm (x,y,z)=[6.16 6.21 4.93], diff<1mm;

end % i
end

%% Gauss: start, fit, output
function [Center_, FWHM_, R2_] = Gauss(id, Pos, Rdg, K, visPlot);
Center_ = zeros(1,K);
FWHM_= zeros(1,K);
R2_ = zeros(1,K);

for i = 1:K
iPos = Pos (id == i);
iRdg = Rdg (id == i);
visPlot(2) =1; if visPlot(2) == 1  figure(i*100); plot (iPos, iRdg, 'b-*'); hold on; end; %xlabel ('Position (mm)'); ylabel (' Charge(relative) '); title ('Source - Target(i)');   grid on; set(gcf, 'units','mm', 'position', [i+3, i+1, 30, 15]); ylim([-10 60]); hold on; end

% initial value estimated
[iRdgMax, idMax] = max(iRdg);  % use iT(iC=max) as the initial guess for center;  Max = center
iPosMax = iPos(idMax);

[Center, FWHM, R2] = fit_Gauss (iPos, iRdg, 50, iPosMax, 2, 2);     % x y a b c d

Center_(i) = Center; %Center(i);
FWHM_(i)= FWHM; %FWHM LGP: 4mm (x,y,z)=[6.16 6.21 4.93], diff<1mm;
R2_(i) = R2; %R2(i);
end % i
end % func
function [Center, FWHM, R2] = fit_Gauss(x, y, a, b, c, d)

 % wiki Gauss function   https://en.wikipedia.org/wiki/Gaussian_function. HWHM = 2.35482c, Area = ac*(2pi)^0.5, 
 
ft_Gauss = fittype ('a/c * exp(-0.5*((x-b)/c)*((x-b)/c)) + d');    % https://www.originlab.com/index.aspx?go = Products/Origin/DataAnalysis/CurveFitting

[f1, gof] = fit (x, y, ft_Gauss, 'start', [a, b, c, d]);

a = f1.a;   % the height of the curve's peak
b = f1.b;   % middle point, also max
c = f1.c;   % = standard deviation, Gaussian RMS width, FWHM = 2sqrt(2ln2) = 2.35482c.  
d = f1.d;   % tail?

Center = b;
FWHM = 2.35482*c;
% peakHeight = a / c;

R2 = gof.rsquare;

% https://stackoverflow.com/questions/52794561/matlab-area-under-Gaussian-curve-trapzy-and-trapzx-y-function-which-is-m

%% plot
% if visPlot (2) == 1; plotLinePoint (i, f1, b ); end 
plotLinePoint (i, f1, b)
function plotLinePoint (i, f1, b)
    PointSize = 12;
    TextSize = 12;
    LineWidth = 2;      

    plot(f1,'-m');  % plot the fitted curve

    plot (b, f1(b), 'ro', 'MarkerSize', PointSize, 'Linewidth', LineWidth);  % peak point
    textStringL = sprintf( '(%s %5.2f)', 'Fitted Center: ', b);
    text(98, 25, textStringL, 'FontSize', TextSize);
    
%      textShiftXL = 0.22;      textShiftYL = 4;
%      textShiftXR = 0.0;     textShiftYR = 4;
%          
%     x11 = b - 0.17/2;   y11 = f1.d;
%     x21 = b + 0.17/2;   y21 = f1.d;
%         
%     xTextL = x11 - textShiftXL;      yTextL = y11 - textShiftYL;
%     xTextR = x21 - textShiftXR;      yTextR = y21- textShiftYR;
%     
%     plot (x11, y11, 'ro', 'MarkerSize', PointSize,'Linewidth', LineWidth);  % equvelant left point
%     textStringL = sprintf( '(%5.2f, %5.2f)', x11, y11 );
%     text(xTextL, yTextL , textStringL, 'FontSize', TextSize);
%         
%     plot (x21, y21, 'ro', 'MarkerSize', PointSize,'Linewidth', LineWidth);  % equvelenat right point
%     textStringR = sprintf( '(%5.2f, %5.2f)', x21, y21);
%     text(xTextR, yTextR, textStringR, 'FontSize', TextSize);
%     
%     x1Line = xline(x11, '--c', 'Linewidth', LineWidth); 
%     x2Line = xline(x21, '--c', 'Linewidth', LineWidth);
%     
%     xlabel ('Time (sec)'); ylabel (' Current (relative) '); hold on;
end
end % fit
function [Final] = output_Gauss(dataFile, t4mmIn_, t4mmOut_, rsquareIn_, rsquareOut_, visPlot);

formatTime = 'yyyymmdd-HHMMSS'; 
timeStamp = datestr(now, formatTime);
folderName = 'C:\Users\coder\Music';        % 'C:\Users\coder\OneDrive - NYU Langone Health\at1GK_Transient-Team\Data_final\Result';
baseFileName = [dataFile, timeStamp, '_4mm.txt'];    % /t does NOT work in .cvs.  good for .xls , as well .txt (fast, but need copy/past to excel)
fullFileName = fullfile (folderName, baseFileName); %
fileID = fopen(fullFileName, 'w');      % https://www.mathworks.com/help/matlab/ref/fopen.html.  w = new file writing, w+ = new file writing and reading, a = appending
fprintf(fileID, '%s \n\n', fullFileName);
fprintf(fileID, '\r');


if contains (dataFile, '669')

fprintf(fileID, '%s \t%s \t%s  \n', 'ID', '4mmIn', '4mmOut');  
fprintf(fileID, 't4mm(sec) = \t%6.4f \t%6.4f \n', t4mmIn_, t4mmOut_);
fprintf(fileID, 'R-Square = \t%6.4f \t%6.4f \n', rsquareIn_, rsquareOut_);
fprintf(fileID, '\n');

elseif contains (dataFile, '16mm')

ave_t4mmIn_ = mean(t4mmIn_);  std_t4mmIn_ = std(t4mmIn_);
ave_t4mmOut_ = mean(t4mmOut_);  std_t4mmOut_ = std(t4mmOut_);
ave_rs4mmIn_ = mean(rsquareIn_);  std_rsquareIn_ = std(rsquareIn_);
ave_rs4mmOut_ = mean(rsquareOut_);  std_rsquareOut_ = std(rsquareOut_);


fprintf(fileID, 'Group ID               \t%5d \t%5d \t%5d \t%5d \t%5d \t%5d \t%5d \t%5d \t%5d \t%5d \t%s \t%s\n', [1 2 3 4 5 6 7 8 9 10], 'ave', 'std'); 
fprintf(fileID, 't4mmIn(sec) = \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \n', t4mmIn_, ave_t4mmIn_, std_t4mmIn_ );
fprintf(fileID, 't4mmOut(sec) = \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \n', t4mmOut_, ave_t4mmOut_, std_t4mmOut_ );
fprintf(fileID, '\n');
fprintf(fileID, 'rsquareIn_ = \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \n', rsquareIn_, ave_rs4mmIn_, std_rsquareIn_ );
fprintf(fileID, 'rsquareOut_(sec) = \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \n', rsquareOut_, ave_rs4mmOut_, std_rsquareOut_ );
fprintf(fileID, '\n');

end

fclose(fileID);


if visPlot (3)  ==1;     
    saveFigures;        
end;     % save Figures and close Figures

    function saveFigures
    figHandles = findall(groot, 'Type', 'figure');    % figHandles = Figure (102), ... all opened figures;
    figNum = sort ( [figHandles.Number], 'ascend');
    for j = 1:length(figNum)
        jfigHandle = figHandles(j);    % findall (gcf, 'Type', 'figure');     % figHandles = findall(gcf, 'Type', 'figure');        % figHandles = Figure (202).  only the current (activate) figure;
        jfigHandleNum= jfigHandle.Number;
        
        baseFigName = [dataFile, 'Fig', num2str(jfigHandleNum), '_', timeStamp, '.png'];
        fullFigName = fullfile (folderName, baseFigName);
        saveas(jfigHandle, fullFigName);
        
        baseFigName1 = [dataFile, 'Fig', num2str(jfigHandleNum), '_', timeStamp, '.fig'];
        fullFigName1 = fullfile (folderName, baseFigName1);
        saveas(jfigHandle, fullFigName1);
    %     close jfigHandle;  % have to close the active gcf and find the next one and it become active.  should be 
    end
    end

Final='Eureka Gauss';

end % output .txt

%% Sigmoid
function [tBetweenShots_, shutterIn_, shutterOut_, tDwell_, tEffect_, tDiff_, tDiffPct_, rsquareIn_, rsquareOut_, heightIn_, heightOut_] = start_Sigmoid (id, Time, Current, cellAll, visPlot)

% from cell array to array
tMidIn = cellAll{1}; tMidOut = cellAll{2}; L = cellAll{3}; R = cellAll{4}; b = cellAll{5}; nominalShotTime = cellAll{6}; collimator = cellAll{7};

% hold memory position, efficiency
if contains(collimator, "4mm 8mm 16mm"); k = 3; else k=10; end;

tBetweenShots_ = zeros(1,k);
xMidIn_ = zeros(1,k); x1In_= zeros(1,k); x2In_=zeros(1,k); shutterIn_=zeros(1,k); rsquareIn_=zeros(1,k); heightIn_=zeros(1,k);
xMidOut_ = zeros(1,k); x1Out_= zeros(1,k); x2Out_=zeros(1,k); shutterOut_=zeros(1,k); rsquareOut_=zeros(1,k); heightOut_=zeros(1,k);
tDwell_= zeros(1,k); tEffect_= zeros(1,k); tDiff_= zeros(1,k); tDiffPct_= zeros(1,k);


for i = 1:length(tMidIn)     %  4mm 8mm 16mm (exclude 4mm in and 4mm out)
    iT = Time (id == i);
    iC = Current (id == i);
    iIndexHalf = ceil (length(iT)/2);  % ceil = round to the nearest integer.   the index of half curve.  Sigmoid fit is for in and out seperately;
    iIndexUp = ceil (iIndexHalf * ( 1 - 0.3));  % need only about 1/3 * [6 6 9] of the High signals
    iIndexDn = ceil (iIndexHalf * ( 1 + 0.3));
    if visPlot (1)  ==1;     figure(i);  plot(iT, iC,'o-');  set(gcf, 'units','centimeters', 'position', [i, i+1, 30, 15]); end;

    % shutter in(up)
    iindexStart = 1;   % unique 
    T = iT (iindexStart : iIndexUp);
    C = iC (iindexStart :iIndexUp);
    
    % plot, QA
    titleSpr = sprintf( collimator );
     if visPlot (1)  ==1; figure(i*100+1);    plot(T, C, 'b*');  xlabel ('Time (sec)'); ylabel (' Current (relative) '); title (titleSpr); grid on; set(gcf, 'units','centimeters', 'position', [i+3, i+1, 30, 15]);  hold on;     end;
    
     % initial guess
    Li = L (i);
    Ri = R (i);
    ixMidIn = tMidIn(i); 
    
    % fitting
    [xMid, x1In, x2In, shutterIn, mdl, gof] = fit_Sigmoid(i, collimator, T, C, Li, Ri, b, ixMidIn, visPlot);  % up = in, dn = out
    
    % fitting results
    xMidIn_(i) = xMid; x1In_(i) = x1In; x2In_(i) = x2In; shutterIn_(i) = shutterIn; rsquareIn_(i) = gof.rsquare;  %  rs = r square    for output
    heightIn = abs(mdl.L - mdl.R);   % fitted L and H.  heightUp = height for in/up transient  
    heightIn_(i) = heightIn;
    
   
    % shutter out (dn)
    iIndexEnd = length(iT);
    T = iT(iIndexDn : iIndexEnd);
    C = iC(iIndexDn : iIndexEnd);
    if visPlot (1)  ==1;     figure(i*100+2); plot(T, C, 'b*'); xlabel ('Time (sec)'); ylabel (' Current (relative) '); title (collimator); grid on; set(gcf, 'units','centimeters', 'position', [i+6, i+1, 30, 15]); hold on;   end;
    
    % initial guess for out / down, Left is high, Right is low;
    Li = R (i);
    Ri = L (i);
    ixMidOut = tMidOut(i);
    
    % fitting
    [xMid, x1Out, x2Out, shutterOut, mdl, gof] = fit_Sigmoid(i, collimator, T, C, Li, Ri, b, ixMidOut ,visPlot);
    
    % fitting results
    xMidOut_(i) = xMid;     x1Out_(i) = x1Out;  x2Out_(i) = x2Out;  shutterOut_(i) = shutterOut;    rsquareOut_(i) = gof.rsquare;
    heightOut = abs(mdl.L - mdl.R);
    heightOut_(i) = heightOut;
    
    % taly
   
    tBetweenShots_(1) = 0;
    if i>1      tBetweenShots_(i) =  xMidIn_(i) - xMidOut_(i-1); end  % 2.2 sec 4mm and 8mm, ? for 16mm;
    tDwell_(i) = x1Out - x2In;
    tEffect_(i) = (shutterIn + shutterOut)/2 + tDwell_(i);
    tDiff_(i) = tEffect_(i) - nominalShotTime(i); % 9 sec for 16mm
    tDiffPct_(i) = tDiff_(i) / nominalShotTime(i)*100;  
end
 
end
function [xMid, x1, x2, shutter, mdl, gof] = fit_Sigmoid(i, collimator, x, y, Lini, Rini, b, xMid, visPlot)
%https://www.mathworks.com/matlabcentral/answers/505984-fitting-a-Sigmoid-curve-using-curve-fitting-tool-box

ft_Sigmoid = fittype('L + (R-L)/(1 + exp(-b*(x-xMid)))', 'indep', 'x'); % ft = fittype('L + (R-L)/(1 + exp(-4*log(3)*(x-xmid)/xscale80))', 'indep', 'x')        % L = lower asymptote, R = Upper asympotote, xmid = 50% point, 

[mdl, gof] = fit (x, y, ft_Sigmoid, 'start', [Lini, Rini, b, xMid ]);   % parameters startpoints for L, R, b, xmid (in the order).  example  [.5 , 3, 15, 3], 16mm [0, 120, 30, 21], 4mm [0, 45, 30, 3.8], 8mm[0, 96, 30, 12]
L = mdl.L;
R = mdl.R;
b = mdl.b;
xMid = mdl.xMid;
yMid = mdl(xMid);

slopeMid = b/4 * (R-L);    % the tangent line: slope y' = k *Sigmoid (1-Sigmoid), at middle, x-xmid = 0, Sigmoid = .5, (1-Sigmoid) = .5, so y' = k / 4;
ySlope0 = yMid - slopeMid*xMid;     % line y = y' x + y0,  x = (y - y0) / y'

x1 = (L - ySlope0) / slopeMid;
x2 = (R - ySlope0) / slopeMid;
shutter = x2 - x1;

visPlot (1)  = 1;    % overide the global setting of 0;
if visPlot (2)  == 1; plotLinePoint (i, collimator, mdl, xMid, yMid, x1, L, x2, R, slopeMid, ySlope0); end;
function plotLinePoint(i, collimator, mdl, xMid, yMid, x1, L, x2, R, slopeM, ySlope);
        PointSize = 12;
        TextSize = 12;
        LineWidth = 2;      
        
       
        if contains (collimator, "669")
        j = 1;
        elseif  contains (collimator,"4mm")
        j=1;
       elseif  contains(collimator,"8mm")
        j=2;
        elseif contains(collimator,"16mm")
         j = 3; 
        end
              
        
        % fitted line, original points, middle point
        plot(mdl, '-m');    % fitted curve
        
        plot(xMid, yMid, 'ro', 'MarkerSize', PointSize,'Linewidth', LineWidth);     % middle point
        xlabel ('Time (sec)'); ylabel (' Current (relative) '); hold on;

        textShiftXMidUp = [0.4, 0.5, 0.6] ;
        textShiftXMidDn = [0.1, 0.1, 0.1] ;
        textShiftXLR = [0.25, 0.25, 0.25];
        textShiftYLR = [3, 5, 6];
        
        if L < R
        xTextMid = xMid - textShiftXMidUp(j);    yTextMid = yMid;
        xTextL = x1 - textShiftXLR(j);    yTextL = L - textShiftYLR(j);
        xTextR = x2 - textShiftXLR(j);    yTextR = R + textShiftYLR(j);
        else 
        xTextMid = xMid + textShiftXMidDn(j); yTextMid = yMid;
        xTextL = x1 - textShiftXLR(j);    yTextL = L + textShiftYLR(j);
        xTextR = x2 - textShiftXLR(j);    yTextR = R - textShiftYLR(j);
        end

        textStringMid = sprintf( '(%5.2f, %5.2f)', xMid, yMid);
        text(xTextMid, yTextMid, textStringMid, 'FontSize', TextSize);
        
        % slope line
        xSlope = linspace(x1, x2, 20);
        ySlope = slopeM * xSlope+ ySlope;
        plot(xSlope, ySlope, '-g', 'linewidth', 2); %https://www.mathworks.com/matlabcentral/answers/44468-plotting-a-linear-equation
        
        % cross section point (x1, L)
        plot(x1, L, 'ro','MarkerSize',PointSize,'Linewidth', LineWidth);
        textStringL = sprintf( '(%5.2f, %5.2f)', x1, L);
        text(xTextL, yTextL, textStringL, 'FontSize', TextSize);
        
        % cross section point (x2, R)
        plot(x2, R, 'ro','MarkerSize',PointSize,'Linewidth', LineWidth);
        textStringR = sprintf( '(%5.2f, %5.2f)', x2, R);
        text(xTextR, yTextR, textStringR, 'FontSize', TextSize);
        
        % vertical and horizontal lines   https://www.mathworks.com/help/matlab/ref/xline.html
        x1Line = xline(x1, '--c', 'Linewidth', LineWidth); 
        x2Line = xline(x2, '--c', 'Linewidth', LineWidth);
%         y1Line = yline(L, '--m',  'Linewidth', LineWidth);
%         y1Line = yline(R, '--m',  'Linewidth', LineWidth);
        
        
        
end

end
function [Final] = output_Sigmoid_all (dataFile, tBetweenShots_, shutterIn_, shutterOut_, tDwell_, tEffect_, tDiff_, tDiffPct_, rsquareIn_, rsquareOut_, heightIn_, heightOut_, visPlot)



formatTime = 'yyyymmdd-HHMMSS'; 
timeStamp = datestr(now, formatTime);
folderName = 'C:\Users\coder\Music';        % 'C:\Users\coder\OneDrive - NYU Langone Health\at1GK_Transient-Team\Data_final\Result';
baseFileName = [dataFile, timeStamp, '.txt'];    % /t does NOT work in .cvs.  good for .xls , as well .txt (fast, but need copy/past to excel)
fullFileName = fullfile (folderName, baseFileName); %
fileID = fopen(fullFileName, 'w');      % https://www.mathworks.com/help/matlab/ref/fopen.html.  w = new file writing, w+ = new file writing and reading, a = appending

fprintf(fileID, '%s \n\n', fullFileName);

    timerError_ = shutterIn_+shutterOut_;
    nominalShotTime_ = [6.0 6.0 9.0];
    
if contains (dataFile, '669')
    
    timerErrorPct_ = timerError_ ./ nominalShotTime_ * 100;
    
fprintf(fileID, '\r');
fprintf(fileID, '%s \t%s \t%s \t%s \n', 'Collimator', '4mm', '8mm', '16mm');  
fprintf(fileID, 'betweenShots (sec) =  \t%6.3f \t%6.3f \t%6.3f \n', tBetweenShots_);
fprintf(fileID, 'shutterIn (sec) =  \t%6.3f \t%6.3f \t%6.3f \n', shutterIn_);
fprintf(fileID, 'shutterOut (sec) = \t%6.3f \t%6.3f \t%6.3f \n', shutterOut_);
fprintf(fileID, 'TimerError (sec) = \t%6.3f \t%6.3f \t%6.3f \n', timerError_);
fprintf(fileID, 'T_Error/nominal(%%) = \t%6.1f%% \t%6.1f%% \t%6.1f%% \n', timerErrorPct_);

fprintf(fileID, 'tDwell (sec) = \t%6.3f \t%6.3f \t%6.3f \n', tDwell_);
fprintf(fileID, 'tEffect (sec) = \t%6.3f \t%6.3f \t%6.3f \n', tEffect_);
fprintf(fileID, 'tDifference(sec) = \t%6.3f \t%6.3f \t%6.3f \n', tDiff_);
fprintf(fileID, 'tDifference (%%) = \t%6.1f%% \t%6.1f%% \t%6.1f%% \n', tDiffPct_);
fprintf(fileID, 'heightIn (mA) =  \t%6.3f \t%6.3f \t%6.3f \n', heightIn_);
fprintf(fileID, 'heightOut (mA) =  \t%6.3f \t%6.3f \t%6.3f \n', heightOut_);

fprintf(fileID, '\r');
fprintf(fileID, 'R-SquareIn = \t%6.3f \t%6.3f \t%6.3f \n', rsquareIn_);
fprintf(fileID, 'R-SquareOut = \t%6.3f \t%6.3f \t%6.3f \n', rsquareOut_);
fprintf(fileID, '\n');

else    % 4mm or 8mm or 16mm

     if contains(dataFile, '4mm') 
         nominalShotTime = nominalShotTime_(1); 
     elseif contains(dataFile, '8mm') 
         nominalShotTime = nominalShotTime_(2);
     elseif contains(dataFile, '16mm') 
         nominalShotTime = nominalShotTime_(3);
     end
    
tBetweenShots_without1st = tBetweenShots_;  % temp array for ave cal; this array has 9 numbers;
tBetweenShots_without1st(1) = []; % exclude the first one which is invalid; 
ave_tBetweenShots_ = mean(tBetweenShots_without1st);  std_tBetweenShots_=std(tBetweenShots_without1st);  

ave_shutterIn_ = mean(shutterIn_);  std_shutterIn_ = std(shutterIn_);
ave_shutterOut_ = mean(shutterOut_);  std_shutterOut_ = std(shutterOut_);
% timerError_ = shutterIn_ + shutterOut_;
ave_timerError_ = mean(timerError_); std_timerError_ = std(timerError_); %https://stats.stackexchange.com/questions/117741/adding-two-or-more-means-and-calculating-the-new-standard-deviation

timerErrorPct_ = timerError_ ./ nominalShotTime * 100;
ave_timerErrorPct_ = mean(timerErrorPct_); std_timerErrorPct_ = std(timerErrorPct_);

ave_tDwell_ = mean(tDwell_);  std_tDwell_ = std(tDwell_);
ave_tEffect_ = mean(tEffect_);  std_tEffect_ = std(tEffect_);
ave_tDiff_ = mean(tDiff_);  std_tDiff_ = std(tDiff_);
ave_tDiffPct_ = mean(tDiffPct_ );  std_tDiffPct_ = std (tDiffPct_);
ave_rsquareIn_ = mean(rsquareIn_ ); std_rsquareIn_ = std(rsquareIn_ );
ave_rsquareOut_ = mean (rsquareOut_  ); std_rsquareOut_ = std(rsquareOut_ );


fprintf(fileID, '\r');
fprintf(fileID, 'Group ID               \t%5d \t%5d \t%5d \t%5d \t%5d \t%5d \t%5d \t%5d \t%5d \t%5d \t%s \t%s\n', [1 2 3 4 5 6 7 8 9 10], 'ave', 'std'); 
fprintf(fileID, 'betweenShots(sec) = \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \n', tBetweenShots_, ave_tBetweenShots_, std_tBetweenShots_);

fprintf(fileID, 'shutterIn(sec) = \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \n', shutterIn_, ave_shutterIn_, std_shutterIn_ );
fprintf(fileID, 'shutterOut(sec) = \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \n', shutterOut_, ave_shutterOut_, std_shutterOut_ );
fprintf(fileID, 'timerError(sec) = \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \n', timerError_, ave_timerError_, std_timerError_);
fprintf(fileID, 'T_Error/nominal(%%) = \t%6.1f%% \t%6.1f%% \t%6.1f%% \t%6.1f%% \t%6.1f%% \t%6.1f%% \t%6.1f%% \t%6.1f%% \t%6.1f%% \t%6.1f%% \t%6.1f%% \t%6.1f%% \n', timerErrorPct_, ave_timerErrorPct_, std_timerErrorPct_ );

fprintf(fileID, 'tDwell(sec) =      \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \n', tDwell_, ave_tDwell_, std_tDwell_ );
fprintf(fileID, 'tEffect(sec)  = \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \n', tEffect_, ave_tEffect_, std_tEffect_ );
fprintf(fileID, 'tDifference(sec) = \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \n', tDiff_, ave_tDiff_, std_tDiff_  );
fprintf(fileID, 'tDifference(%%) = \t%6.1f%% \t%6.1f%% \t%6.1f%% \t%6.1f%% \t%6.1f%% \t%6.1f%% \t%6.1f%% \t%6.1f%% \t%6.1f%% \t%6.1f%% \t%6.1f%% \t%6.1f%% \n', tDiffPct_, ave_tDiffPct_, std_tDiffPct_ );
fprintf(fileID, '\r');
fprintf(fileID, 'R-SquareIn =   \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \n', rsquareIn_, ave_rsquareIn_, std_rsquareIn_);
fprintf(fileID, 'R-SquareOut =   \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \t%6.3f \n', rsquareOut_, ave_rsquareOut_, std_rsquareOut_);
fprintf(fileID, '\n');

end

fclose(fileID);


if visPlot (3)  ==1;     saveFigures;        end;     % save Figures and close Figures
    function saveFigures
    figHandles = findall(groot, 'Type', 'figure');    % figHandles = Figure (102), ... all opened figures;
    figNum = sort ( [figHandles.Number], 'ascend');
    for j = 1:length(figNum)
        jfigHandle = figHandles(j);    % findall (gcf, 'Type', 'figure');     % figHandles = findall(gcf, 'Type', 'figure');        % figHandles = Figure (202).  only the current (activate) figure;
        jfigHandleNum= jfigHandle.Number;
        
        baseFigName = [dataFile, 'Fig', num2str(jfigHandleNum), '_', timeStamp, '.png'];
        fullFigName = fullfile (folderName, baseFigName);
        saveas(jfigHandle, fullFigName);
        
        baseFigName1 = [dataFile, 'Fig', num2str(jfigHandleNum), '_', timeStamp, '.fig'];
        fullFigName1 = fullfile (folderName, baseFigName1);
        saveas(jfigHandle, fullFigName1);
    %     close jfigHandle;  % have to close the active gcf and find the next one and it become active.  should be 
    end
    end

Final='Eureka Sigmoid'; 
end
% print(figure(1),'-dpng','-r300','Monthly_QA_Timer_current.tif')