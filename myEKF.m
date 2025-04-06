function [X_Est, P_Est, GT] = myEKF(out, q)
    %% Sensor Data Extraction from Simulation Output
    accel = squeeze(permute(out.Sensor_ACCEL.signals.values, [3, 2, 1]));
    gyro  = squeeze(permute(out.Sensor_GYRO.signals.values, [3, 2, 1]));
    mag   = squeeze(permute(out.Sensor_MAG.signals.values, [3, 2, 1]));

    % Extract ToF ranges and status flags
    tof_front        = squeeze(out.Sensor_ToF2.signals.values(:,1,:));
    tof_front_status = squeeze(out.Sensor_ToF2.signals.values(:,4,:));

    tof_left         = squeeze(out.Sensor_ToF3.signals.values(:,1,:));
    tof_left_status  = squeeze(out.Sensor_ToF3.signals.values(:,4,:));

    tof_right        = squeeze(out.Sensor_ToF1.signals.values(:,1,:));
    tof_right_status = squeeze(out.Sensor_ToF1.signals.values(:,4,:));

    pos     = out.GT_position.signals.values;   % Nx3 ground truth
    rotQuat = out.GT_rotation.signals.values;   % Nx4 quaternions
    timeVec = squeeze(out.GT_position.time);             % Nx1

    sensor_calibration = load("sensor_calibration.mat");

    %% Assume everything is the same length
    N = length(pos);

    GT = pos;  % store ground-truth positions for reference

    %% Frame Convention
    accel = (sensor_calibration.Rotw_a * accel')';
    gyro  = (sensor_calibration.Rotw_a * gyro')';
    mag   = (sensor_calibration.Rotw_mag * mag')';

    %% Sensor Calibration
    accel = accel - sensor_calibration.accel_bias;
    gyro  = gyro  - sensor_calibration.gyro_bias;
    mag   = (mag - sensor_calibration.b_mag) * sensor_calibration.A_mag;


    mean(accel(:,1));
    mean(accel(:,2)); % Y-acceleration, which is forward.
    mean(accel(:,3));
    figure(1)
    plot(timeVec,accel(:,1),timeVec,accel(:,2),timeVec,accel(:,3))
    title('accel raw readings')


    %% Convert GT quaternion to Euler angles
    % By default, quat2eul returns [yaw, pitch, roll] in ZYX order,
    % but you want to treat eulAngles(:,1) as yaw
    eulAngles = quat2eul(rotQuat);  % [yaw pitch roll] in ZYX
    yawGT     = eulAngles(:,1) +pi;
    GT = [GT, yawGT];

    %% EKF Initialization (6D)
    % State = [ x, y, vx, vy, psi, dpsi ]
    X_Est = zeros(N, 6);
    P_Est = cell(N,1);

    % Use GT for initial position and yaw, zero velocity
    x0   = pos(1,1);
    y0   = pos(1,2);
    psi0 = yawGT(2); 
    X_Est(1,:) = [ x0, y0, 0, 0, psi0, 0 ];
    P_Est{1}   = diag([0.01, 0.01, 0.05, 0.05, 0.001, 0.05]);

    % Process noise Q
    %Q = diag([1e-6, 1e-6, 1e-3, 1e-3, 1e-5, 1e-3]);
    Q = q;

    % Base measurement noise (without status scaling)
    scaleFactors = [1, 0.1, 0.1, 0.1];
    RdiagBase = [
        sensor_calibration.R_psi, ...
        sensor_calibration.R_tof_left, ...
        sensor_calibration.R_tof_middle, ...
        sensor_calibration.R_tof_right
    ];

    %% Main EKF Loop
    for k = 1:N-1
        dt = timeVec(k+1) - timeVec(k);
        if dt <= 0, dt = 1e-2; end

        xk = X_Est(k,:)';

        % Prediction
        [x_pred, F] = predictionStepSymbolic(xk, accel(k,:), gyro(k,:), dt);
        P_pred = F * P_Est{k} * F' + Q;

        % Build measurement
        [z_meas, H] = measurementModelOptim(x_pred, mag(k,:), ...
                                            tof_right(k),tof_front(k),tof_left(k));

        %% --- Dynamically scale the ToF measurement noise ---
        % row2 -> right, row3 -> front, row4 -> left
        frontScale = interpretToFStatus(tof_front_status(k));
        leftScale  = interpretToFStatus(tof_left_status(k));
        rightScale = interpretToFStatus(tof_right_status(k));

        localScales = [1e2, rightScale, frontScale, leftScale]; %dont trust the magnetometer
        R_local = diag(localScales .* (scaleFactors .* RdiagBase));

        % EKF Update
        [x_upd, P_upd] = updateStep(x_pred, P_pred, z_meas, H, R_local);

        % Store
        X_Est(k+1,:) = x_upd';
        P_Est{k+1}   = P_upd;

    end
    
    mag_yaw = wrapToPi(atan2(mag(:,2), mag(:,1)));
    figure(2)
    plot(timeVec, mag_yaw, 'r'); hold on;
    plot(timeVec, yawGT, 'k'); legend('Magnetometer yaw', 'GT yaw');

end

%% ==============================================================
function [x_pred, F] = predictionStepSymbolic(xk, accel, gyro, dt)
    % xk(1) = x, xk(2) = y
    % xk(3) = vx, xk(4) = vy
    % xk(5) = psi, xk(6) = dpsi

    x   = xk(1);
    y   = xk(2);
    vx  = xk(3);
    vy  = xk(4);
    psi = xk(5);
    dpsi= xk(6);

    % We'll assume accel(2) is forward acceleration in your chosen frame
    af  = accel(2);
    gyZ = gyro(3);

    c = cos(psi);
    s = sin(psi);

    x_pred = zeros(6,1);


     % If forward = +Y in world frame,
    x_pred(1) = x + vx*dt + 0.5 * s * af * dt^2;
    x_pred(2) = y + vy*dt + 0.5 * c * af * dt^2;
    x_pred(3) = vx + s * af * dt;
    x_pred(4) = vy + c * af * dt;

    x_pred(5) = psi + dpsi*dt;                   % yaw
    x_pred(6) = gyZ;                           % yaw rate from gyro


    % Jacobian (6x6)
    F = eye(6);

    F(1,3) = dt;
    F(1,5) = 0.5 * c * af * dt^2;
    F(2,5) =  -0.5 * s * af * dt^2;
    
    F(3,5) = c * af * dt;
    F(4,5) =  s * af* dt;

end

%% ==============================================================
function [z_meas, H] = measurementModelOptim(x_pred, mag, d1, d2, d3)
    % We measure yaw from magnetometer plus 3 ToF distances
    psi_meas = atan2(mag(2), mag(1));

    % Build z_meas
    z_meas = [psi_meas; d1; d2; d3];

    % predicted yaw = x_pred(5)
    psi = x_pred(5);
    x   = x_pred(1);
    y   = x_pred(2);

    offsets = [-pi/2, 0, pi/2];
    angles  = psi + offsets;

    d_pred = zeros(3,1);
    H = zeros(4,6);  % 4 measurements, 6 states

    for i = 1:3
        theta = angles(i);
        [t, ~, dt_dx, dt_dy, dt_dpsi] = rayBoxIntersection(x, y, theta);
        d_pred(i) = t;
        row = i + 1;  % rows 2..4 in z_meas
        H(row,1) = dt_dx;     % partial wrt x
        H(row,2) = dt_dy;     % partial wrt y
        H(row,5) = dt_dpsi;   % partial wrt psi
    end

    % row 1 => yaw partial
    % yaw_meas depends only on psi => partial(psi_meas)/partial(psi_est) = 1
    H(1,5) = 1;
end

%% ==============================================================
function [x_upd, P_upd] = updateStep(x_pred, P_pred, z_meas, H, R)
    % Predicted measurement
    psi_est = x_pred(5);
    x_est   = x_pred(1);
    y_est   = x_pred(2);

    offsets =  [-pi/2, 0, pi/2];
    angles  = psi_est + offsets;
    d_pred  = zeros(3,1);

    for i = 1:3
        d_pred(i) = rayBoxIntersection(x_est, y_est, angles(i));
    end

    z_pred = [psi_est; d_pred];

    % innovation
    y_tilde = z_meas - z_pred;

    % wrap yaw difference
    y_tilde(1) = wrapToPi(y_tilde(1));

    % Compute Kalman gain
    S = H * P_pred * H' + R;
    K = P_pred * H' / S;

    % update state
    x_upd = x_pred + K * y_tilde;
    % wrap final yaw
    x_upd(5) = wrapToPi(x_upd(5));

    % update covariance
    I = eye(length(x_pred));
    P_upd = (I - K * H) * P_pred;
end

%% ==============================================================
function [t, branch, dt_dx, dt_dy, dt_dpsi] = rayBoxIntersection(x, y, theta)
    x_left   = -1.2; x_right = 1.2;
    y_bottom = -1.2; y_top   = 1.2;

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
        t     = t_v;
        branch= 1;
        dt_dx = -1/cth;
        dt_dy = 0;
        dt_dtheta = (x_wall - x) * sth / (cth^2);
    else
        t     = t_h;
        branch= 2;
        dt_dx = 0;
        dt_dy = -1/sth;
        dt_dtheta = - (y_wall - y) * cth / (sth^2);
    end
    dt_dpsi = dt_dtheta;
end

%% ---------------------------------------------------------------
function scale = interpretToFStatus(statusVal)
    % status = 0 => reading is fine
    % status = 2 => quite unreliable
    % status = 4 => very bad, do not trust
    switch statusVal
        case 0
            scale = 1.0;
        case 2
            scale = 10.0;
        case 4
            scale = 1e6;
        otherwise
            scale = 10.0;  % fallback for unexpected codes
    end
end