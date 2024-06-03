% Clear workspace, close figures, and clear command window
clear all
close all
clc

%------------------Global variables to ensure exchange between functions
global Mthr_instance;       % ULFS stages - M threshold settings
global fthr_instance;       % ULFS stages - f threshold settings
global UFLS_stages;         % ULFS stages - load decrease 
global fLIM;                % Frequency limit for M calculation
global f_store;             % array of f values from numerical calculations
global M_store;             % array of M values from numerical calculations
global T_store;             % array of t values from numerical calculations
global f_M_function;        % Anonymous fthr(M) tripping function
global f_rated;             % rated frequency value [Hz]

%------------------Initialization of global variables
f_store       = [];
M_store       = [];
T_store       = [];

%------------------UFLS settings
fthr_instance = [49.0  48.8  48.6  48.4  48.2  48.0];
Mthr_instance = [2.0   1.80  1.60  1.40  1.20  1.00];
UFLS_stages   = [0.10  0.10  0.10  0.10  0.10  0.10];
fLIM          = 47.5;

%------------------Anonymous fthr(M) tripping function
% 1 = No UFLS
% 2 = Conventional UFLS
% 3 = L-shaped fthr(M) function
% 4 = ellipse-shaped fthr(M) function

UFLS_type = 3;

if     UFLS_type == 1
    f_M_function = @(M, fset, Mset) (M>=0) .* 0;
elseif UFLS_type == 2
    f_M_function = @(M, fset, Mset) (M>=0) .* fset;
elseif UFLS_type == 3
    f_M_function = @(M, fset, Mset) (M<=Mset) .* fset + (M>Mset) .* fLIM;
elseif UFLS_type == 4
    f_M_function = @(M, fset, Mset) fLIM + sqrt(max(0, (1 - (M.^2) ./ (Mset^2)) .* ((fset - fLIM)^2)));
end

%------------------Simulation duration and timestep
simulation_duration = 10;      % seconds
simulation_timestep = 0.001;   % seconds

%------------------Generate time vector for simulation
time = 0 : simulation_timestep : simulation_duration;

%------------------Define initial power imbalance with a step function u
u = zeros(size(time));         % Initialization
K = -0.70;                     % Step amplitude
u(time >= 1) = K;              % Step at t=1
udot = [0, u(2:end)-u(1:end-1)]/simulation_timestep; % time derivative of u

%------------------Frequency Response Model (FRM) parameters
H  = 4;           % Inertia constant
D  = 1;           % Damping constant
FH = 0.3;         % Feedback gain
R  = 0.05;        % Droop
KM = 0.95;        % Feedback loop gain
TR = 8;           % Time constant of the feedback loop
f_rated = 50;     % Rated system frequency [Hz]

%------------------Simulink results
r_simulink = sim('SFRM_with_UFLS_simulink.slx');

%------------------Direct FRM branch - swing equation
num_F1 = [1];                  % numerator
den_F1 = [2 * H, D];           % denominator
F1 = tf(num_F1, den_F1);       % create transfer function

%------------------Feedback FRM branch - frequency control
num_FB1 = [FH * TR * KM, KM];  % numerator
den_FB1 = [TR * R, R];         % denominator
FB1 = tf(num_FB1, den_FB1);    % create transfer function

%------------------Overall FRM transfer function
G = F1 / (1 + F1 * FB1);       % FRM transfer function

%------------------Convert G transfer function to state-space
[num_G, den_G] = tfdata(G, 'v');
[A,B,C,D] = tf2ss(num_G,den_G);
sys = ss(A,B,C,D);

%------------------Initial conditions
x0 = zeros(3,1);

%------------------Simulate the step response using transfer function approach
Gu  = lsim(sys, u, time, x0);

%------------------Solve the system using numerical integration (ode45)
options = odeset('MaxStep', simulation_timestep); % integration time step
[t_ss, rez_ss] = ode45(@(t,x) equations(t,x,A,B,C,D,u,udot,time), [0 simulation_duration], x0, options);
y_ss = C * rez_ss';   % calculate output variable (FRM frequency) using state variables

%------------------Plot the results
figure;

subplot(2,2,1);
plot(t_ss, f_rated+f_rated*y_ss, 'LineWidth', 3); grid on; hold on;
plot(r_simulink.tout, r_simulink.frequency, 'LineWidth', 5, 'LineStyle', ':');
plot(time, f_rated+f_rated*Gu, 'LineWidth', 4); 
ylim([47, 50]);
legend('Numeric Integration', 'Simulink', 'Without UFLS (pure FRM Transfer Function)');
xlabel('Time [sec]')
ylabel('Frequency [Hz]')
title('Frequency with respect to time', 'FontSize', 16)

subplot(2,2,3);
plot(T_store, M_store, 'LineWidth', 3); grid on; hold on;
plot(r_simulink.tout, r_simulink.M, 'LineWidth', 5, 'LineStyle', ':');
xlim([0, simulation_duration]);
ylim([0, 10]);
legend('Numeric Integration', 'Simulink');
xlabel('Time [sec]')
ylabel('M [sec]')
title('Frequency stability margin', 'FontSize', 16)

subplot(2,2,[2,4]);
plot(M_store, f_store, 'LineWidth', 3); grid on; hold on;
plot(r_simulink.M, r_simulink.frequency, 'LineWidth', 5, 'LineStyle', ':');
x_fm = 0:0.01:5;
for indx = 1 : length(fthr_instance)
    y_fm = f_M_function(x_fm, fthr_instance(indx), Mthr_instance(indx));
    plot(x_fm, y_fm, 'k');
end
xlim([0, 5]);
ylim([fLIM, f_rated]);
legend('Numeric Integration', 'Simulink');
xlabel('M [sec]')
ylabel('Frequency [Hz]')
title('f-M plane', 'FontSize', 16)

%------------------Function to be solved by numerical integration (ode45)
function xdot = equations(t, x, A, B, C, D, u, udot, time)
    global f_store; 
    global M_store; 
    global T_store; 
    global f_rated;

    % Monitoring previous value of UFLS throughout numerical integration
    persistent u_feedback_prev;

    % Initialization of UFLS for first numerical integration step
    if isempty(u_feedback_prev) || t == 0
        u_feedback_prev = 0;
    end    

    % setting up initial power deficit   
    u_set0 = interp1(time, u, t, 'previous');
    udot0  = interp1(time, udot, t, 'previous');
    
    % calculating the system output and its derivative in CURRENT state
    u_current    = u_set0           + u_feedback_prev;
    xdot_current = A * x            + B * u_current;
    y_current    = C * x            + D * u_current; 
    ydot_current = C * xdot_current + D * udot0;
    
    % calculate frequency and its derivative in CURRENT state
    f    = f_rated * y_current + f_rated;
    dfdt = f_rated * ydot_current;
    
    % call UFLS function
    [u_feedback, M] = UFLS(f, dfdt, u_feedback_prev);
    
    % update the previous value of u_feedback
    u_feedback_prev = u_feedback;
    
    % add feedback to initial power deficit
    u_set = u_set0 + u_feedback;
    
    % calculate system equations
    xdot = A * x + B * u_set;
    
    % store to global variables
    f_store(end+1) = f;
    M_store(end+1) = M;
    T_store(end+1) = t;
end

%------------------UFLS function
function [u_feedback, M] = UFLS(f, dfdt, prev_u_feedback)
    global Mthr_instance;
    global fthr_instance;
    global UFLS_stages;
    global fLIM;
    global f_M_function;

    % settings of UFLS
    f_thrs       = fthr_instance;   % frequency thresholds in Hz
    thrs_sizes   = UFLS_stages;     % corresponding stage sizes 
    M_thrs       = Mthr_instance;   % M thresholds in sec

    % setting limits to dfdt (always negative)
    if dfdt>-1e-15
        dfdt_lim = -1e-15;
    elseif dfdt<-1e+9
        dfdt_lim = -1e+9;
    else
        dfdt_lim = dfdt;
    end
    
    % calculate M
    M = (fLIM-f)/dfdt_lim;

    % current status of u_feedback
    u_status = prev_u_feedback; 

    % UFLS activation
    for idx = 1:length(f_thrs)    % examine all UFLS stages        
        trip_threshold = f_M_function(M, f_thrs(idx), M_thrs(idx)); % Calculate M-dependent f threshold
        if f <= trip_threshold & (u_status < sum(thrs_sizes(1:idx))) % if threshold violated and stage not activated yet
            u_status = sum(thrs_sizes(1:idx)); % activate stage
        end
    end

    u_feedback = u_status; % assign to UFLS output
end

