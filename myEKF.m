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
    accel2 = (sensor_calibration.Rotw_a2 * accel2')';  % APPLY CORRECT ROTATION FOR SECOND ACCELEROMETER
    
    accel = accel - sensor_calibration.accel_bias;
    gyro  = gyro  - sensor_calibration.gyro_bias;
    mag   = (mag - sensor_calibration.b_mag) * sensor_calibration.A_mag;
    accel2 = accel2 - sensor_calibration.accel2_bias;  % SUBTRACT SECOND ACCELEROMETER BIAS
    
    %% --- NEW: FILTER EACH ACCELEROMETER AND FUSE THEIR X-AXIS ---
    % FILTER PRIMARY accelerometer (sampling rate 104 Hz)
    fs1 = 104;
    fc1 = 1;  % cutoff frequency in Hz (tune as needed)
    [b1,a1] = butter(2, fc1/(fs1/2));
    accel1_smoothed = filtfilt(b1,a1, accel(:,1));
    
    % FILTER SECOND accelerometer (sampling rate 100 Hz)
    fs2 = 100;
    fc2 = 1;  % cutoff frequency in Hz
    [b2,a2] = butter(2, fc2/(fs2/2));
    accel2_smoothed = filtfilt(b2,a2, accel2(:,1));
    
    % FUSE using weighted average.
    % IF you have variance estimates, set w1 = 1/var1 and w2 = 1/var2.
    % Here we assume equal weights:
    w1 = 2; w2 = 1;
    accel_fused = (w1*accel1_smoothed + w2*accel2_smoothed) / (w1 + w2);
    
    % USE THE FUSED X-AXIS acceleration to correct the magnetometer X reading.
    % (The idea is that interference in the magnetometer X channel is partly due to vehicle vibrations.)
    scale = 0.00002;  % Hyperparameter (tune this based on your system)
    mag(:,1) = mag(:,1) - scale * accel_fused;
    % -----------------------------------------------------------
    
    %% DEBUG: Plot corrected magnetometer and fused accel data
    figure;
    subplot(2,1,1);
    plot(mag(:,1)); title('Corrected Magnetometer X');
    subplot(2,1,2);
    plot(accel_fused); title('Fused Smoothed Accel X');
    
    %% Convert GT quaternion to Euler angles
    eulAngles = quat2eul(rotQuat);  % returns [yaw, pitch, roll] in ZYX order
    yawGT = eulAngles(:,1) + pi;      % Add π offset if needed
    GT = [GT, yawGT];
    
    %% EKF Initialization (6D: [x, y, vx, vy, psi, dpsi])
    X_Est = zeros(N, 6);
    P_Est = cell(N,1);
    x0 = pos(1,1);
    y0 = pos(1,2);
    % IMPORTANT: USE THE SECOND VALUE FOR YAW (since the first is NaN)
    psi0 = yawGT(2);
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
        [x_pred, F] = predictionStepSymbolic(xk, accel(k,:), gyro(k,:), dt);
        P_pred = F * P_Est{k} * F' + Q;
        
        % Build measurement vector
        [z_meas, H] = measurementModelOptim(x_pred, mag(k,:), ...
                                             tof_right(k), tof_front(k), tof_left(k));
        
        %% --- Dynamically scale the ToF measurement noise ---
        frontScale = interpretToFStatus(tof_front_status(k));
        leftScale  = interpretToFStatus(tof_left_status(k));
        rightScale = interpretToFStatus(tof_right_status(k));
        localScales = [1e2, rightScale, frontScale, leftScale];  % moderate scaling
        R_local = diag(localScales .* (scaleFactors .* RdiagBase));
        %% -----------------------------------------------------------
        
        % EKF Update
        [x_upd, P_upd] = updateStep(x_pred, P_pred, z_meas, H, R_local);
        
        % Store updated state and covariance
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
    
    % Assume accel(2) is forward acceleration (if forward is +Y)
    af = accel(2);
    gyZ = gyro(3);
    
    c = cos(psi);
    s = sin(psi);
    
    x_pred = zeros(6,1);
    % Update positions (note: forward direction now defined by our coordinate system)
    x_pred(1) = x + vx*dt + 0.5 * s * af * dt^2;
    x_pred(2) = y + vy*dt + 0.5 * c * af * dt^2;
    % Update velocities
    x_pred(3) = vx + s * af * dt;
    x_pred(4) = vy + c * af * dt;
    % Yaw update
    x_pred(5) = psi + dpsi*dt;
    % Yaw rate update from gyro
    x_pred(6) = gyZ;
    
    % Jacobian F (6x6)
    F = eye(6);
    F(1,3) = dt;
    F(1,5) = 0.5 * c * af * dt^2;
    F(2,4) = dt;
    F(2,5) = -0.5 * s * af * dt^2;
    F(3,5) = c * af * dt;
    F(4,5) = s * af * dt;
    F(5,6) = dt;
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
