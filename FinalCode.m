%% Managment plan 1 - Basic Model Output (w/o fishing) 

param.h = 10;       % Factor for max.  ingestion rate g^1/4/yr
param.Fc = 0.2;     % Critical feeding level 
param.eps = 0.7;    % Assimilation eff. 
param.epsR = 0.55;  % Reproductive eff. 
param.Mort = .8     % Mortality Factor  g^-1/4/yr
param.y = 1.5;      % Factor for clearance rate  L/yr/g
param.Wo = 0.001;   % Weight of offspring g
param.Wm = 137;     % Weight at mature is 100

param.K = 12; 
param.r = .85;      % Resource growth rate 

param.k = param.Fc*param.h;

tRange = [0:1000];   % Time frame

F = 0;

% Grid Values
param.nGrid = 50; 
w = logspace(log10(param.Wo), log10(param.Wm), param.nGrid);
param.deltaw = gradient(w); %The center of the grid

% Lets run the model:
[t,Y] = Manag_one(tRange, param, w, F);


n = Y(:,1:end-2) ;   % split the solution into components - Immature / sum( n .* dw )
N = Y(:,end-1);      % split the solution into components - Adults / NUMBERS!!!
R = Y(:,end);        % split the solution into components - Resources

%% Managment plan 2 - Basic Model Output (w/o fishing) 


param.h = 10;       % Factor for max.  ingestion rate g^1/4/yr
param.Fc = 0.2;     % Critical feeding level 
param.eps = 0.7;    % Assimilation eff. 
param.epsR = 0.55;  % Reproductive eff. 
param.Mort = .8     % Mortality Factor  g^-1/4/yr
param.y = 1.5;      % Factor for clearance rate  L/yr/g
param.Wo = 0.001;   % Weight of offspring g
param.Wm = 137;     % Weight at mature is 100

param.K = 12; 
param.r = .85;      % Resource growth rate 

param.k = param.Fc*param.h;

tRange = [0:1000];   % Time frame

F = 0;

% Grid Values
param.nGrid = 50; 
w = logspace(log10(param.Wo), log10(param.Wm), param.nGrid);
param.deltaw = gradient(w); %The center of the grid

% Lets run the model:
[t,Y] = Manag_two(tRange, param, w, F);

n2 = Y(:,1:end-2) ;   % split the solution into components - Immature / sum( n .* dw )
N2 = Y(:,end-1);      % split the solution into components - Adults / NUMBERS!!!
R2 = Y(:,end);        % split the solution into components - Resources

%% Nice tiled plots of basic model output 

% Managment option 1 and 2 will have the same basic output due to fishing
% being set to zero. 

figure;
tl = tiledlayout(2,1);

nexttile;

hold on
yyaxis left
plot(t,log10(R),'DisplayName','Resource','LineWidth',3);
ylabel('log(N) [indv.]')
yyaxis right
plot(t,log10(N),'DisplayName','Adults','LineWidth',3);
ylabel('log(R) [g/L]')
title('(A) Adult and resource biomass over time');
legend('Location','NorthWest')
hold off
xlabel('Time [years]')

nexttile;


ss = zeros;

for i = 1:1001
    ss(i) = sum(log10(n(i,:)));
   
end

hold on
yyaxis left
plot(t, ss,'c-','LineWidth',3,'DisplayName','Sum of Juvis');
ylabel('Cumulative log(n) [#/L/g]')
yyaxis right
plot(t, log10(N), 'LineWidth',3,'DisplayName','Adults','LineWidth',3);
xlabel('Time [years]');
ylabel(' log(N) [indv.]')
legend('Location','NorthWest')
title('(B) Adult and total juvenile biomass over time');


%% Nice plot for Managment Type

figure;
tl = tiledlayout(2,1);
nexttile;

syms w 

    a = 1;
    b = 3;
    min=0;
    max=1;
    Fw = 40;
    
    f = min + ((max-min) .* ((1 ./(1+exp(-b.*(w-Fw)))))); 

fplot(f,[0 137],'LineWidth',3)
xlabel('Herring size (g)');
ylabel('Fishing Selectivity Factor');
title('(A) Managment 1 Option: Minimum Size');

nexttile

syms w 

wp = 70 ;    % prefered weight?
O = .3 ;     % width function
beta = 1     % ratio

f = exp(-(log(w/(beta*wp)))^2/(O^2))

fplot(f,[0 137],'c-','LineWidth',3)
ylim([0 1])
xlabel('Herring size (g)');
ylabel('Fishing Selectivity Factor');
title('(B) Managment 2 Option: Weight Window');



%% Cumul. juv. + Adults / managem. scenario 1

param.h = 10;       % Factor for max.  ingestion rate g^1/4/yr
param.Fc = 0.2;     % Critical feeding level 
param.eps = 0.7;    % Assimilation eff. 
param.epsR = 0.55;  % Reproductive eff. 
param.Mort = 0.8;   % Mortality Factor  g^-1/4/yr
param.y = 1.5;      % Factor for clearance rate  L/yr/g
param.Wo = 0.001;   % Weight of offspring g
param.Wm = 137;     % Weight at mature is 100

param.K = 12; 
param.r = .85;      % Resource growth rate 

param.k = param.Fc*param.h;
tRange = [0:500];   % Time frame

% Grid Values
param.nGrid = 50; 
w = logspace(log10(param.Wo), log10(param.Wm), param.nGrid);
param.deltaw = gradient(w); %The center of the grid

% figure;
% plotting
hold on
i = 1;

for F = [0.5 , 1]
    [t1,Y1] = Manag_one(tRange, param, w, F);
    
    n1 = Y1(:,1:end-2) ;   % split the solution into components - Immature / sum( n .* dw )
    N1 = Y1(:,end-1);      % split the solution into components - Adults / NUMBERS!!!
    R1 = Y1(:,end);        % split the solution into components - Resources

    ss1 = zeros;
    for j = 1:length(t1)
    ss1(j) = sum(log10(n1(j,:)));
    end

    subplot(1,2,i)
    
    title('Mang. scenario 1/ Fishing effort = '+string(F))
    
    hold on
    yyaxis left 
    plot(t1, log10(ss1),'LineWidth',2.5);
    ylabel('log10(Cumul. juv.) [indv./g/L]')
    ylim([-5, 5]);
    
    
    yyaxis right
    plot(t1, log10(N1),'LineWidth',2.5);
    ylabel('log10(N) [indv.]')
    ylim([-20,15]);
    
    legend('Cum. juv.' , 'Adults');
    xlabel('Time (years)')
    
    i = i + 1; 
end
xlabel('Time (years)')


%% Now Plot Resource. + Adults / managem. scenario 1

param.h = 10;       % Factor for max.  ingestion rate g^1/4/yr
param.Fc = 0.2;     % Critical feeding level 
param.eps = 0.7;    % Assimilation eff. 
param.epsR = 0.55;  % Reproductive eff. 
param.Mort = 0.8;   % Mortality Factor  g^-1/4/yr
param.y = 1.5;      % Factor for clearance rate  L/yr/g
param.Wo = 0.001;   % Weight of offspring g
param.Wm = 137;     % Weight at mature is 100

param.K = 12; 
param.r = .85;      % Resource growth rate 

param.k = param.Fc*param.h;
tRange = [0:500];   % Time frame

% Grid Values
param.nGrid = 50; 
w = logspace(log10(param.Wo), log10(param.Wm), param.nGrid);
param.deltaw = gradient(w); %The center of the grid


hold on
i = 1;

for F = [0.5 , 1]
    [t1,Y1] = Manag_one(tRange, param, w, F);
    
    n1 = Y1(:,1:end-2) ;   % split the solution into components - Immature / sum( n .* dw )
    N1 = Y1(:,end-1);      % split the solution into components - Adults / NUMBERS!!!
    R1 = Y1(:,end);        % split the solution into components - Resources
    
    subplot(3,2,i)
    
    title('Mang. scenario 1/ Fishing effort = '+string(F))
    
    hold on
    yyaxis left 
    plot(t1, log10(R1),'LineWidth',2.5);
    ylabel('log10(R) [g/L]');

    yyaxis right
    plot(t1, log10(N1),'LineWidth',2.5);
    ylabel('log10(N) [indv.]');
    ylim([-20, 20]);
    
    legend('Resource' , 'Adults');
    legend('Location','SouthEast');
    xlabel('Time (years)')
    
    %%%%%%%%%%%% 
    
    %Plotting the respective feeding level - NESTOR PLOT
    subplot(3,2,i+2)
    fw = ((param.y.*w).*R1)./(((param.y.*w).*R1)+(param.h*w.^(3/4)));
    fwend= fw([389, 397, 406],:); %We can use either 19 years or 8 years sections
    plot(log10(w),fwend,'-','LineWidth',2.5)
    hold on
    yline(0.2,'--b',{'Critical feeding level'});
    legend('Year 389','Year 397','Year 406');
    legend('Location','NorthWest');
    xlabel('Weight (log)');
    ylabel('Feeding level');
    hold off
    
    %%%%%%%%%
    % plotting the number of individuals per weight class at specific years
    subplot(3,2,i+4)
    her = n1;
    hend = her([389, 397, 406],:); %selecting the years to analyze
    
    plot(log10(w(1:end-1)),log10(hend),'-','LineWidth', 2.5)
    hold on

    legend('Year 389','Year 397','Year 406');
    title('Density of juveniles per weight class'); 
    xlabel('Weight (log)');
    ylabel('log10(juv.) [indv./g/L]')   
    
    i = i + 1; 
end



%% Than - Cumul. juv. + Adults / managem. scenario 2

param.h = 10;       % Factor for max.  ingestion rate g^1/4/yr
param.Fc = 0.2;     % Critical feeding level 
param.eps = 0.7;    % Assimilation eff. 
param.epsR = 0.55;  % Reproductive eff. 
param.Mort = 0.8;   % Mortality Factor  g^-1/4/yr
param.y = 1.5;      % Factor for clearance rate  L/yr/g
param.Wo = 0.001;   % Weight of offspring g
param.Wm = 137;     % Weight at mature is 100

param.K = 12; 
param.r = .85;      % Resource growth rate 

param.k = param.Fc*param.h;
tRange = [0:500];   % Time frame

% Grid Values
param.nGrid = 50; 
w = logspace(log10(param.Wo), log10(param.Wm), param.nGrid);
param.deltaw = gradient(w); %The center of the grid

% figure;
% plotting
hold on
i = 1;

for F = [0.5 , 1]

    [t2,Y2] = Manag_two(tRange, param, w, F);
    
    n2 = Y2(:,1:end-2) ;   % split the solution into components - Immature / sum( n .* dw )
    N2 = Y2(:,end-1);      % split the solution into components - Adults / NUMBERS!!!
    R2 = Y2(:,end);        % split the solution into components - Resources
    
    ss2 = zeros;
    for j = 1:length(t1)
    ss2(j) = sum(log10(n2(j,:)));
    end

    subplot(1,2,i)
    
    title('Mang. scenario 2/ Fishing effort = '+string(F))
    
    hold on
    yyaxis left 
    plot(t2, log10(ss2),'LineWidth',2.5);
    ylabel('log10(Cumul. juv.) [indv./g/L]')
    ylim([-5, 5]);

    yyaxis right
    plot(t2, log10(N2),'LineWidth',2.5);
    ylabel('log10(N) [indv.]')
    ylim([-20,15]);
    
    legend('Cum. juv.' , 'Adults');
    %  ylim([-18, 3]);
    xlabel('Time (years)')
    
    i = i + 1; 
end
xlabel('Time (years)')
% ylabel('Number of individuals')
% legend(legends);legend('boxoff');


%% Than - Resource. + Adults / managem. scenario 2

param.h = 10;       % Factor for max.  ingestion rate g^1/4/yr
param.Fc = 0.2;     % Critical feeding level 
param.eps = 0.7;    % Assimilation eff. 
param.epsR = 0.55;  % Reproductive eff. 
param.Mort = 0.8;   % Mortality Factor  g^-1/4/yr
param.y = 1.5;      % Factor for clearance rate  L/yr/g
param.Wo = 0.001;   % Weight of offspring g
param.Wm = 137;     % Weight at mature is 100

param.K = 12; 
param.r = .85;      % Resource growth rate 

param.k = param.Fc*param.h;
tRange = [0:500];   % Time frame

% Grid Values
param.nGrid = 50; 
w = logspace(log10(param.Wo), log10(param.Wm), param.nGrid);
param.deltaw = gradient(w); %The center of the grid

% figure;
% plotting
hold on
i = 1;

for F = [0.5 , 1]

    [t2,Y2] = Manag_two(tRange, param, w, F);
   
    n2 = Y2(:,1:end-2) ;   % split the solution into components - Immature / sum( n .* dw )
    N2 = Y2(:,end-1);      % split the solution into components - Adults / NUMBERS!!!
    R2 = Y2(:,end);        % split the solution into components - Resources
    

    subplot(3,2,i)
    
    
    title('Mang. scenario 2/ Fishing effort = '+string(F))
    
    hold on
    yyaxis left 
    plot(t2, log10(R2),'LineWidth',2.5);
    ylabel('log10(R) [g/L]');

    yyaxis right
    plot(t2, log10(N2),'LineWidth',2.5);
    ylabel('log10(N) [indv.]')
    hold off
    
    legend('Resource' , 'Adults');
    legend('Location','SouthEast');
    xlabel('Time (years)')
    
    %%%%%%%%%%%% 
    
    %Plotting the respective feeding level - NESTOR PLOT
    subplot(3,2,i+2)
    fw = ((param.y.*w).*R2)./(((param.y.*w).*R2)+(param.h*w.^(3/4)));
    %fwend= fw(446:19:484,:)
    fwend= fw([371, 388, 398],:); %We can use either 19 years or 8 years sections
    plot(log10(w),fwend,'-','LineWidth',2.5)
    hold on
    yline(0.2,'--b',{'Critical feeding level'});
    legend('Year 371','Year 388','Year 398');
    legend('Location','NorthWest');
    xlabel('Weight (log)');
    ylabel('Feeding level');
    hold off
    
    %%%%%%%%%
    %plotting the number of individuals per weight class at specific years
    subplot(3,2,i+4)
    her = n2;
    hend = her([371, 388, 398],:); %selecting the years to analyze
    plot(log10(w(1:end-1)),log10(hend),'-','LineWidth',2.5)
    hold on
    legend('Year 371','Year 388','Year 398');
   
    
    legend('Year 389','Year 397','Year 406');
    title('Density of juveniles per weight class'); 
    xlabel('Weight (log)');
    ylabel('log10(juv.) [indv./g/L]')   

    hold off
    
    i = i + 1; 
end

%% Direct Comparison of Managment Senarios

param.h = 10;       % Factor for max.  ingestion rate g^1/4/yr
param.Fc = 0.2;     % Critical feeding level 
param.eps = 0.7;    % Assimilation eff. 
param.epsR = 0.55;  % Reproductive eff. 
param.Mort = 0.8;     % Mortality Factor  g^-1/4/yr
param.y = 1.5;      % Factor for clearance rate  L/yr/g
param.Wo = 0.001;   % Weight of offspring g
param.Wm = 137;     % Weight at mature is 100

param.K = 12; 
param.r = .85;      % Resource growth rate 

param.k = param.Fc*param.h;

tRange = [0:500];   % Time frame

% Grid Values
param.nGrid = 50; 
w = logspace(log10(param.Wo), log10(param.Wm), param.nGrid);
param.deltaw = gradient(w); %The center of the grid


legends = [];
i = 1;
j = 1;

for F = 0.5
    [t1,Y1] = Manag_one(tRange, param, w, F);
    [t2,Y2] = Manag_two(tRange, param, w, F);
    
    n1 = Y1(:,1:end-2) ;   % split the solution into components - Immature / sum( n .* dw )
    N1 = Y1(:,end-1);      % split the solution into components - Adults / NUMBERS!!!
    R1 = Y1(:,end);        % split the solution into components - Resources
    
    n2 = Y2(:,1:end-2) ;   % split the solution into components - Immature / sum( n .* dw )
    N2 = Y2(:,end-1);      % split the solution into components - Adults / NUMBERS!!!
    R2 = Y2(:,end);        % split the solution into components - Resources
    
    
    ss1 = zeros;
    for i = 1:length(t1)
    ss1(i) = sum(log10(n1(i,:)));
    end

    
    ss2 = zeros;
    for i = 1:length(t2)
    ss2(i) = sum(log10(n2(i,:)));
    end
      
    subplot(3, 1, 1)
    plot(t1, log10(R1),'m-','LineWidth',1.5);
    hold on 
    plot(t2, log10(R2),'g-','LineWidth',1.5);
    hold off
    xlabel('Time (years)')
    ylabel('log10(R) [g/L]');
    legend('Manag. scenario 1: Resource', 'Manag. scenario 2: Resource')
    legend('Location','SouthEast');
    title('Fishing effort = 0.5')

    subplot(3, 1, 2)
    hold on 
    plot(t1, log10(N1),'m-','LineWidth',1.5);
    plot(t2, log10(N2),'g-','LineWidth',1.5);
    xlabel('Time (years)')
    ylabel('log10(N) [indv.]')
    legend('Manag. scenario 1: Adults', 'Manag. scenario 2: Adults')
    legend('Location','SouthEast');
    title('Fishing effort = 0.5')

    subplot(3, 1, 3)
    hold on 
    plot(t1, log10(ss1),'m-','LineWidth',1.5);
    plot(t2, log10(ss2),'g-','LineWidth',1.5);
    xlabel('Time (years)')
    ylabel('log10(Cumul. juv.) [indv./g/L]')
    legend('Manag. scenario 1: Cumul. juv.', 'Manag. scenario 2: Cumul. juv.')
    legend('Location','SouthEast');
    title('Fishing effort = 0.5')

    xlabel('Time (years)')
end
    
xlabel('Time (years)')

%% Managment plan 1

function [t,Y] = Manag_one(tRange, param, w, F)

n0 = [ones(1,param.nGrid)*20];
R0 = 1;

% Run model
%
[t, Y] = ode45(@SizemodelDeriv, tRange, [n0 R0], [], param);

    % ---------------------------------------------------------
    % Derivative function
    % ---------------------------------------------------------
    function dYdt = SizemodelDeriv(t,Y,param)
        %
        
        n = Y(1:param.nGrid);
        n = [n]'; 
 
        % NOTE:
        % babbies/juvis are n(1:end-1)
        % adults are n(end)
                
        R = Y(param.nGrid+1);
        R = [R]';
        
 %  ------------ HERRING (Prey) ------------ %
        
  % calc feeding level
  
        fw = ((param.y.*w).*R)./(((param.y.*w).*R)+(param.h*w.^(3/4)));
        
  % Now available energy and growth, reporduction, and mortality

        Ea_w = param.eps*param.h*(fw-param.Fc).*w.^(3/4); % calc energy available.
        Ea_w(fw<param.Fc) = 0; % 0 if f(w) <= fc
        
        g = Ea_w; %calc growth
        
        mort = param.Mort.*w.^(-1/4); % Mortality 
        
  %  ------------ Fluxes for Herring ------------ %

    % Only have advection, no diffusion 

    % Advective fluxes
    
                ix = 2:(param.nGrid);
                
                J(ix) = g(ix-1).*n(ix-1); % all the cells one into the next
                
                J(1) = param.epsR*Ea_w(end)*n(end)/param.Wo; % left boundary (becoming babies)-> Reproduction
               
                J(param.nGrid+1) = g(end-1).*n(end-1); % right boundary (flux of juvis becoming adults)
        

    % Calc fishing on Herring 
    
    b = 3;
    min=0;
    max=1;
    Fw = 34;
    
    f = min + ((max-min) .* ((1 ./(1+exp(-b.*(w(1:end)-Fw)))))); %Anna edited
    
    % Calc. Derivative for Herring

    % Growth (calc above)       -  Natural Mortality  -   Fishing
    dNadt =   g(end-1)*n(end-1)   -  mort(end)*n(end) - f(end).*F .* n(end); % adults/mature (also a #)
    
    
    dndt = -(J(2:(param.nGrid)) - J(1:param.nGrid-1))./param.deltaw(1:end-1) - mort(1:end-1).* n(1:end-1) - f(1:end-1).*F .* n(1:end-1); % # of indv.  

 %  ------------ ZOOPLANKTON (Resource) ------------ %
 
    Cmax = param.h*w.^(3/4);
        
    x = param.r*R*(1-R/param.K);
    y = sum(fw(1:end-1).*Cmax(1:end-1).*n(1:end-1).*param.deltaw(1:end-1)) + fw(end).*Cmax(end).*n(end);

    dRdt = x - y ;
    
    
%    fishing mortality(w) = selectivity(w) * F0

 %  ------------ End: put it all together ------------ %

    % Make dPdt a column vector:
    dYdt = [dndt,dNadt,dRdt]';

    end

end

%% Managment plan 2 

function [t,Y] = Manag_two(tRange, param, w, F)

n0 = [ones(1,param.nGrid)*20];
R0 = 1;

% Run model
%
[t, Y] = ode45(@SizemodelDeriv, tRange, [n0 R0], [], param);

    % ---------------------------------------------------------
    % Derivative function
    % ---------------------------------------------------------
    function dYdt = SizemodelDeriv(t,Y,param)
        %
        
        n = Y(1:param.nGrid);
        n = [n]'; 
 
        % NOTE:
        % babbies/juvis are n(1:end-1)
        % adults are n(end)
                
        R = Y(param.nGrid+1);
        R = [R]';
        
 %  ------------ HERRING (Prey) ------------ %
        
  % calc feeding level
  
        fw = ((param.y.*w).*R)./(((param.y.*w).*R)+(param.h*w.^(3/4)));
        
  % Now available energy and growth, reporduction, and mortality

        Ea_w = param.eps*param.h*(fw-param.Fc).*w.^(3/4); % calc energy available.
        Ea_w(fw<param.Fc) = 0; % 0 if f(w) <= fc
        
        g = Ea_w; %calc growth
        
        mort = param.Mort.*w.^(-1/4); % Mortality 
        
  %  ------------ Fluxes for Herring ------------ %

    % Only have advection, no diffusion 

    % Advective fluxes
    
                ix = 2:(param.nGrid);
                
                J(ix) = g(ix-1).*n(ix-1); % all the cells one into the next
                
                J(1) = param.epsR*Ea_w(end)*n(end)/param.Wo; % left boundary (becoming babies)-> Reproduction
               
                J(param.nGrid+1) = g(end-1).*n(end-1); % right boundary (flux of juvis becoming adults)
        

    % Calc fishing on Herring 
    
    wp = 70 ;   
    O = .45 ;     
    beta = 1.2;     

   f = exp(-(log(w(1:end)./(beta*wp))).^2./(O^2));
    
    % Calc. Derivative for Herring

    % Growth (calc above)       -  Natural Mortality  -   Fishing
    dNadt =   g(end-1)*n(end-1)   -  mort(end)*n(end) - f(end).*F .* n(end); % adults/mature (also a #)
    
    
    dndt = -(J(2:(param.nGrid)) - J(1:param.nGrid-1))./param.deltaw(1:end-1) - mort(1:end-1).* n(1:end-1) - f(1:end-1).*F .* n(1:end-1); % # of indv.  

 %  ------------ ZOOPLANKTON (Resource) ------------ %
 
    Cmax = param.h*w.^(3/4);
        
    x = param.r*R*(1-R/param.K);
    y = sum(fw(1:end-1).*Cmax(1:end-1).*n(1:end-1).*param.deltaw(1:end-1)) + fw(end).*Cmax(end).*n(end);

    dRdt = x - y ;
    
    
%    fishing mortality(w) = selectivity(w) * F0

 %  ------------ End: put it all together ------------ %

    % Make dPdt a column vector:
    dYdt = [dndt,dNadt,dRdt]';

    end

end
