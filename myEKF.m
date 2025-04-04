function [X_Est, P_Est, GT] = myEKF(out)
    %% Sensor Data Extraction from Simulation Output
    % Extract sensor readings from Simulink output structure 'out'
    accel = squeeze(permute(out.Sensor_ACCEL.signals.values, [3, 2, 1]));
    gyro  = squeeze(permute(out.Sensor_GYRO.signals.values, [3, 2, 1]));
    mag   = squeeze(permute(out.Sensor_MAG.signals.values, [3, 2, 1]));
    tof_front = squeeze(out.Sensor_ToF1.signals.values(:,1,:));
    tof_left  = squeeze(out.Sensor_ToF2.signals.values(:,1,:));
    tof_right = squeeze(out.Sensor_ToF3.signals.values(:,1,:));
    pos   = out.GT_position.signals.values;
    timeVec = out.GT_position.time;
    sensor_calibration = load("sensor_calibration.mat");

    %% Time Alignment and Truncation
    % Ensure all sensor signals are aligned in time and have consistent length
    N = min([size(accel,1), size(gyro,1), size(mag,1), length(tof_front), length(timeVec)]);
    pos = pos(1:N,:);
    timeVec = timeVec(1:N);
    GT = pos; % ground truth position

    %% Frame Convention

    accel = (sensor_calibration.Rotw_a * accel')';
    gyro = (sensor_calibration.Rotw_a * gyro')';
    mag = (sensor_calibration.Rotw_mag * mag')';

    %% TODO: Accel2 load and calibration


    
    %% Sensor Calibration
    % Rotate and scale magnetometer to align with world frame and normalize strength
    accel = accel - sensor_calibration.accel_bias;
    gyro = gyro - sensor_calibration.gyro_bias;
    mag = (mag - sensor_calibration.b_mag) * sensor_calibration.A_mag;


    %% EKF Initialization
    % Define initial state: [x y vx vy psi dpsi gyro_bias accel_bias]
    X_Est = zeros(N, 8);
    P_Est = cell(N,1);
    X_Est(1,:) = [ pos(1,1:2), 0, 0, 0, 0, 0, 0 ];
    P_Est{1} = diag([0.01, 0.01, 0.1, 0.1, 0.01, 0.01, 0.001, 0.001]);

    % Define process and measurement noise matrices
    Q = diag([1e-6, 1e-6, 1e-3, 1e-3, 1e-6, 1e-3, 1e-8, 1e-8]);
    R = diag([(1*pi/180)^2, 0.05^2, 0.05^2, 0.05^2]);

    %% Main EKF Loop
    for k = 1:N-1
        dt = timeVec(k+1) - timeVec(k);
        if dt <= 0, dt = 1e-2; end  % handle zero or negative time step

        % Prediction Step: Uses symbolic expressions for state transition and Jacobian
        xk = X_Est(k,:)';
        [x_pred, F] = predictionStepSymbolic(xk, accel(k,:), gyro(k,:), dt);

        % Measurement Step: Uses optimization to estimate position (x, y) from ToF sensors
        [z_meas, H] = measurementModelOptim(x_pred, mag_corrected(k,:), tof_front(k), tof_left(k), tof_right(k));

        % Kalman Filter Update
        [x_upd, P_upd] = updateStep(x_pred, P_Est{k}, z_meas, H, R);
        X_Est(k+1,:) = x_upd';
        P_Est{k+1} = P_upd;
    end
end

function [x_pred, F] = predictionStepSymbolic(xk, accel, gyro, dt)
    %% Symbolic Constant Acceleration Prediction Model with Analytical Jacobian
    % Declare symbolic variables for state and intermediate terms
    persistent F_template
    if isempty(F_template)
        syms x y vx vy psi dpsi gyro_bias accel_bias dt af gy real
        c = cos(psi); s = sin(psi);
        
        % Prediction model equations
        f1 = x + vx*dt + 0.5*c*af*dt^2;
        f2 = y + vy*dt + 0.5*s*af*dt^2;
        f3 = vx + c*af*dt;
        f4 = vy + s*af*dt;
        f5 = psi + dpsi*dt;
        f6 = gy;
        f7 = gyro_bias;
        f8 = accel_bias;
        
        f = [f1; f2; f3; f4; f5; f6; f7; f8];
        X = [x; y; vx; vy; psi; dpsi; gyro_bias; accel_bias];
        F_template = matlabFunction(jacobian(f, X), 'Vars', {x, y, vx, vy, psi, dpsi, gyro_bias, accel_bias, dt, af});
    end

    % Evaluate numerical state
    x = xk(1); y = xk(2); vx = xk(3); vy = xk(4);
    psi = xk(5); dpsi = xk(6); gb = xk(7); ab = xk(8);
    af = accel(3) - ab;
    gy = gyro(3) - gb;
    c = cos(psi); s = sin(psi);

    % Compute predicted state using manually written expressions
    x_pred = zeros(8,1);
    x_pred(1) = x + vx*dt + 0.5*c*af*dt^2;
    x_pred(2) = y + vy*dt + 0.5*s*af*dt^2;
    x_pred(3) = vx + c*af*dt;
    x_pred(4) = vy + s*af*dt;
    x_pred(5) = psi + dpsi*dt;
    x_pred(6) = gy;
    x_pred(7) = gb;
    x_pred(8) = ab;

    % Evaluate symbolic Jacobian with current state
    F = F_template(x, y, vx, vy, psi, dpsi, gb, ab, dt, af);
end

function [z_meas, H] = measurementModelOptim(x_pred, mag, d1, d2, d3)
    %% Measurement model using ToF sensor triangulation by optimization
    % Estimate (x, y) location based on ToF residual minimization
    psi = x_pred(5);
    psi_meas = atan2(mag(1), mag(3));
    pos_est = locate_cart_optimization(d1, d2, d3, psi);
    x = pos_est(1); y = pos_est(2);

    % Measurement vector: [yaw_measured; tof1; tof2; tof3]
    z_meas = [psi_meas; d1; d2; d3];

    % Jacobian H: partial derivatives of measurements wrt state (approximate)
    H = zeros(4,8);
    H(1,5) = 1;     % yaw measurement wrt psi
    H(2:4,1:2) = [1 0; 0 1; 0 1]; % simplified pos influence on tof
end

function [x_upd, P_upd] = updateStep(x_pred, P_pred, z_meas, H, R)
    %% EKF Update Step
    z_pred = [x_pred(5); x_pred(1:2); x_pred(2)];  % simplified prediction
    y_tilde = z_meas - z_pred;
    S = H * P_pred * H' + R;
    K = P_pred * H' / S;
    x_upd = x_pred + K * y_tilde;
    P_upd = (eye(length(x_pred)) - K * H) * P_pred;
end

function pos_est = locate_cart_optimization(d_front, d_left, d_right, psi)
    %% Optimization routine to estimate (x,y) based on ToF ray intersections
    offsets = [0, pi/2, -pi/2];
    angles = psi + offsets;
    d = [d_front; d_left; d_right];
    residual = @(pos) tofs_residual(pos, angles, d);
    pos0 = [0, 0]; lb = [-1.2, -1.2]; ub = [1.2, 1.2];
    options = optimoptions('lsqnonlin','Display','off');
    pos_est = lsqnonlin(residual, pos0, lb, ub, options);
end

function r = tofs_residual(pos, angles, d)
    %% Residual function for optimization: predicts sensor distances for given position
    x = pos(1); y = pos(2);
    x_left = -1.2; x_right = 1.2; y_bottom = -1.2; y_top = 1.2;
    r = zeros(length(d),1);
    for i = 1:length(d)
        theta = angles(i);
        if cos(theta) > 0, t_v = (x_right - x)/cos(theta);
        elseif cos(theta) < 0, t_v = (x_left - x)/cos(theta);
        else, t_v = inf;
        end
        if sin(theta) > 0, t_h = (y_top - y)/sin(theta);
        elseif sin(theta) < 0, t_h = (y_bottom - y)/sin(theta);
        else, t_h = inf;
        end
        t_v = max(t_v, 0); t_h = max(t_h, 0);
        r(i) = min(t_v, t_h) - d(i);
    end
end
