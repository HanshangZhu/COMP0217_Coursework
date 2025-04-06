function [X_Est, P_Est, GT] = myEKF(out, q)
    %% Sensor Data Extraction from Simulation Output
    accel = squeeze(permute(out.Sensor_ACCEL.signals.values, [3, 2, 1]));
    gyro  = squeeze(permute(out.Sensor_GYRO.signals.values, [3, 2, 1]));
    mag   = squeeze(permute(out.Sensor_MAG.signals.values, [3, 2, 1]));
    accel2 = squeeze(permute(out.Sensor_LP_ACCEL.signals.values, [3, 2, 1]));  % SECOND ACCELEROMETER

    % Extract ToF ranges and status flags
    tof_front        = squeeze(out.Sensor_ToF2.signals.values(:,1,:));
    tof_front_status = squeeze(out.Sensor_ToF2.signals.values(:,4,:));
    
    tof_left         = squeeze(out.Sensor_ToF3.signals.values(:,1,:));
    tof_left_status  = squeeze(out.Sensor_ToF3.signals.values(:,4,:));
    
    tof_right        = squeeze(out.Sensor_ToF1.signals.values(:,1,:));
    tof_right_status = squeeze(out.Sensor_ToF1.signals.values(:,4,:));
    
    pos     = out.GT_position.signals.values;   % Nx3 ground truth
    rotQuat = out.GT_rotation.signals.values;     % Nx4 quaternions
    timeVec = squeeze(out.GT_position.time);       % Nx1

    sensor_calibration = load("sensor_calibration.mat");
    
    %% Assume everything is the same length
    N = length(pos);
    GT = pos;  % store ground-truth positions for reference
    
    %% Frame Convention & Sensor Calibration
    accel = (sensor_calibration.Rotw_a * accel')';
    gyro  = (sensor_calibration.Rotw_a * gyro')';
    mag   = (sensor_calibration.Rotw_mag * mag')';
    accel2 = (sensor_calibration.Rotw_a2 * accel2')';  % USE CORRECT ROTATION FOR SECOND ACCELEROMETER
    
    accel = accel - sensor_calibration.accel_bias;
    gyro  = gyro - sensor_calibration.gyro_bias;
    mag   = (mag - sensor_calibration.b_mag) * sensor_calibration.A_mag;
    accel2 = accel2 - sensor_calibration.accel2_bias;  % SUBTRACT SECOND ACCELEROMETER BIAS
    
    %% --- NEW: FILTER EACH ACCELEROMETER ALONG BOTH X AND Y, THEN FUSE ---
    % FILTER PRIMARY accelerometer (sampling rate 104 Hz)
    fs1 = 104;
    fc1 = 0.25;  % cutoff frequency in Hz (tune as needed)
    [b1, a1] = butter(2, fc1/(fs1/2));
    accel1_smoothed_x = filtfilt(b1, a1, accel(:,1));
    accel1_smoothed_y = filtfilt(b1, a1, accel(:,2));
    
    % FILTER SECOND accelerometer (sampling rate 100 Hz)
    fs2 = 100;
    fc2 = 0.25;  % cutoff frequency in Hz
    [b2, a2] = butter(2, fc2/(fs2/2));
    accel2_smoothed_x = filtfilt(b2, a2, accel2(:,1));
    accel2_smoothed_y = filtfilt(b2, a2, accel2(:,2));
    
    % FUSE using weighted average.
    % Example weights (you may set these based on measured variances)
    w1 = 2; w2 = 1;
    fused_accel_x = (w1 * accel1_smoothed_x + w2 * accel2_smoothed_x) / (w1 + w2);
    fused_accel_y = (w1 * accel1_smoothed_y + w2 * accel2_smoothed_y) / (w1 + w2);
    % Create a fused acceleration matrix (for later use in the prediction step)
    accel_fused = [fused_accel_x, fused_accel_y, accel(:,3)];  
    % NOTE: We keep the original Z-axis from the primary sensor.
    
    % USE THE FUSED accelerometer X & Y to correct the magnetometer X reading as before.
    scale = 0.0002;  % Hyperparameter (tune as needed)
    mag(:,1) = mag(:,1) - scale * fused_accel_x;
    % -----------------------------------------------------------
    
    %% DEBUG: Plot fused accelerometer and corrected magnetometer
    figure;
    subplot(2,1,1);
    plot(mag(:,1)); title('Corrected Magnetometer X');
    subplot(2,1,2);
    plot(fused_accel_x); title('Fused Smoothed Accel X');

    figure;
    subplot(2,1,1);
    plot(mag(:,2)); title('Corrected Magnetometer Y');
    subplot(2,1,2);
    plot(fused_accel_y); title('Fused Smoothed Accel Y');
    
    
    %% Convert GT quaternion to Euler angles
    eulAngles = quat2eul(rotQuat);  % returns [yaw, pitch, roll] in ZYX order
    yawGT = eulAngles(:,1) + pi;      % Add π offset if needed
    GT = [GT, yawGT];
    
    %% EKF Initialization (6D: [x, y, vx, vy, psi, dpsi])
    X_Est = zeros(N, 6);
    P_Est = cell(N,1);
    x0 = pos(1,1);
    y0 = pos(1,2);
    psi0 = yawGT(2);  % IMPORTANT: use the second sample to avoid NaN
    X_Est(1,:) = [x0, y0, 0, 0, psi0, 0];
    P_Est{1} = diag([0.01, 0.01, 0.05, 0.05, 0.001, 0.05]);
    
    %% Process noise Q supplied as input
    Q = q;
    
    %% Base measurement noise (without dynamic scaling)
    scaleFactors = [1, 0.1, 0.1, 0.1];
    RdiagBase = [ sensor_calibration.R_psi, ...
                  sensor_calibration.R_tof_left, ...
                  sensor_calibration.R_tof_middle, ...
                  sensor_calibration.R_tof_right ];
    
    %% MAIN EKF LOOP
    for k = 1:N-1
        dt = timeVec(k+1) - timeVec(k);
        if dt <= 0, dt = 1e-2; end
    
        xk = X_Est(k,:)';
        [x_pred, F] = predictionStepSymbolic(xk, accel_fused(k,:)', gyro(k,:), dt);
        P_pred = F * P_Est{k} * F' + Q;
    
        [z_meas, H] = measurementModelOptim(x_pred, mag(k,:), ...
                                     tof_right(k), tof_front(k), tof_left(k));
    
        %% --- Build R_local (ToF status scaled) BEFORE Mahalanobis check ---
        frontScale = interpretToFStatus(tof_front_status(k));
        leftScale  = interpretToFStatus(tof_left_status(k));
        rightScale = interpretToFStatus(tof_right_status(k));
        localScales = [10, rightScale, frontScale, leftScale];  % yaw, right, front, left
        R_local = diag(localScales .* (scaleFactors .* RdiagBase));
    
        %% --- Mahalanobis gating on yaw only ---
        psi_meas = z_meas(1);
        psi_pred = x_pred(5);
        yaw_innovation = wrapToPi(psi_meas - psi_pred);
    
        R_psi = R_local(1,1);  % yaw variance (already scaled)
        S_psi = H(1,:) * P_pred * H(1,:)' + R_psi;
        d2_yaw = (yaw_innovation)^2 / S_psi;
    
        if d2_yaw > 9
            H(1,:) = 0;
            R_local(1,1) = 1e12;
            fprintf('Yaw gated at step %d, Mahalanobis distance = %.2f\n', k, d2_yaw);
        end
    
    [x_upd, P_upd] = updateStep(x_pred, P_pred, z_meas, H, R_local);
    X_Est(k+1,:) = x_upd';
    P_Est{k+1} = P_upd;
end

    
    %% Debug: Plot magnetometer yaw (raw) vs. GT yaw
    mag_yaw = wrapToPi(atan2(mag(:,2), mag(:,1)));
    figure(4)
    plot(timeVec, mag_yaw, 'r'); hold on;
    plot(timeVec, wrapToPi(yawGT), 'k');
    legend('Magnetometer yaw', 'GT yaw');
    title('Magnetometer vs Ground Truth Yaw');
    xlabel('Time [s]'); ylabel('Yaw [rad]');

     err_x = X_Est(:,1) - GT(:,1);
        err_y = X_Est(:,2) - GT(:,2);
        err_yaw = wrapToPi(X_Est(:,5) - GT(:,4));  % GT(:,4) is yawGT
        
        figure;
        subplot(3,1,1); plot(timeVec, err_x); title('Error in x');
        subplot(3,1,2); plot(timeVec, err_y); title('Error in y');
        subplot(3,1,3); plot(timeVec, err_yaw); title('Error in yaw');
end

%% ==============================================================
function [x_pred, F] = predictionStepSymbolic(xk, accel, gyro, dt)
    % State xk = [x; y; vx; vy; psi; dpsi]
    x   = xk(1);
    y   = xk(2);
    vx  = xk(3);
    vy  = xk(4);
    psi = xk(5);
    dpsi= xk(6);
    
    % ACCEL: Now using the fused accelerometer reading (in body frame)
    % We first transform the fused acceleration from body frame to world frame.
    % Let a_fused = [a_x; a_y] in body frame.
    a_body = accel(1:2);  % use both X and Y
    % Transformation from body to world (given yaw psi)
    R_bw = [cos(psi), -sin(psi); sin(psi), cos(psi)];
    a_world = R_bw * a_body;  % [a_wx; a_wy]
    
    a_wx = a_world(1);
    a_wy = a_world(2);
    
    % Get gyro Z for yaw rate
    gyZ = gyro(3);
    
    % Predict state using constant acceleration over dt
    x_pred = zeros(6,1);
    x_pred(1) = x + vx*dt + 0.5*a_wx*dt^2;
    x_pred(2) = y + vy*dt + 0.5*a_wy*dt^2;
    x_pred(3) = vx + a_wx*dt;
    x_pred(4) = vy + a_wy*dt;
    x_pred(5) = psi + dpsi*dt;
    x_pred(6) = gyZ;
    
    % Jacobian F (6x6) - for simplicity we use a basic approximation
    % %TODO:Improve model
    F = eye(6);
    F(1,3) = dt;
    F(1,1) = 1; % note: a more detailed Jacobian would include partial derivatives wrt psi due to rotation in a_world.
    F(2,4) = dt;
    F(5,6) = dt;
    % FOR A BETTER MODEL, COMPUTE PARTIALS OF a_world WITH RESPECT TO psi.
    % This is left as an exercise for further tuning.
end

%% ==============================================================
function [z_meas, H] = measurementModelOptim(x_pred, mag, d1, d2, d3)
    % We measure yaw from the magnetometer via atan2.
    psi_meas = atan2(mag(2), mag(1));
    z_meas = [psi_meas; d1; d2; d3];
    
    % Predicted yaw from state
    psi = x_pred(5);
    x   = x_pred(1);
    y   = x_pred(2);
    
    offsets = [-pi/2, 0, pi/2];
    angles = psi + offsets;
    
    d_pred = zeros(3,1);
    H = zeros(4,6);  % 4 measurements x 6 states
    for i = 1:3
        theta = angles(i);
        [t, ~, dt_dx, dt_dy, dt_dpsi] = rayBoxIntersection(x, y, theta);
        d_pred(i) = t;
        row = i + 1;
        H(row,1) = dt_dx;
        H(row,2) = dt_dy;
        H(row,5) = dt_dpsi;
    end
    H(1,5) = 1;
end

%% ==============================================================
function [x_upd, P_upd] = updateStep(x_pred, P_pred, z_meas, H, R)
    psi_est = x_pred(5);
    x_est = x_pred(1);
    y_est = x_pred(2);
    
    offsets = [-pi/2, 0, pi/2];
    angles = psi_est + offsets;
    d_pred = zeros(3,1);
    for i = 1:3
        d_pred(i) = rayBoxIntersection(x_est, y_est, angles(i));
    end
    z_pred = [psi_est; d_pred];
    
    % Innovation
    y_tilde = z_meas - z_pred;
    y_tilde(1) = wrapToPi(y_tilde(1));
    
    S = H * P_pred * H' + R;
    K = P_pred * H' / S;
    
    x_upd = x_pred + K * y_tilde;
    x_upd(5) = wrapToPi(x_upd(5));
    
    I = eye(length(x_pred));
    P_upd = (I - K * H) * P_pred;
end

%% ==============================================================
function [t, branch, dt_dx, dt_dy, dt_dpsi] = rayBoxIntersection(x, y, theta)
    x_left = -1.2; x_right = 1.2;
    y_bottom = -1.2; y_top = 1.2;
    
    cth = cos(theta);
    sth = sin(theta);
    
    if cth > 0
        x_wall = x_right;
    elseif cth < 0
        x_wall = x_left;
    else
        x_wall = NaN;
    end
    
    if ~isnan(x_wall) && abs(cth) > 1e-6
        t_v = (x_wall - x) / cth;
    else
        t_v = inf;
    end
    
    if sth > 0
        y_wall = y_top;
    elseif sth < 0
        y_wall = y_bottom;
    else
        y_wall = NaN;
    end
    
    if ~isnan(y_wall) && abs(sth) > 1e-6
        t_h = (y_wall - y) / sth;
    else
        t_h = inf;
    end
    
    if t_v <= 0, t_v = inf; end
    if t_h <= 0, t_h = inf; end
    
    if t_v < t_h
        t = t_v;
        branch = 1;
        dt_dx = -1/cth;
        dt_dy = 0;
        dt_dtheta = (x_wall - x) * sth / (cth^2);
    else
        t = t_h;
        branch = 2;
        dt_dx = 0;
        dt_dy = -1/sth;
        dt_dtheta = - (y_wall - y) * cth / (sth^2);
    end
    dt_dpsi = dt_dtheta;
end

%% ---------------------------------------------------------------
function scale = interpretToFStatus(statusVal)
    switch statusVal
        case 0
            scale = 1.0;
        case 2
            scale = 100.0;
        case 4
            scale = 1e6;
        otherwise
            scale = 10.0;
    end
end
