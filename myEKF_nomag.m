% myEKF_nomag.m: main EKF functions using the measurement model for update

function [X_Est, P_Est, GT, gyro_z] = myEKF_nomag(out, q, r, plt)
    %% === Sensor Data Extraction ===
    accel  = squeeze(permute(out.Sensor_ACCEL.signals.values, [3, 2, 1]));
    gyro   = squeeze(permute(out.Sensor_GYRO.signals.values, [3, 2, 1]));
    accel2 = squeeze(permute(out.Sensor_LP_ACCEL.signals.values, [3, 2, 1]));

    tof_front        = squeeze(out.Sensor_ToF2.signals.values(:,1,:));
    tof_front_status = squeeze(out.Sensor_ToF2.signals.values(:,4,:));
    tof_left         = squeeze(out.Sensor_ToF3.signals.values(:,1,:));
    tof_left_status  = squeeze(out.Sensor_ToF3.signals.values(:,4,:));
    tof_right        = squeeze(out.Sensor_ToF1.signals.values(:,1,:));
    tof_right_status = squeeze(out.Sensor_ToF1.signals.values(:,4,:));

    pos     = out.GT_position.signals.values;
    rotQuat = out.GT_rotation.signals.values;
    timeVec = squeeze(out.GT_position.time);

    sensor_calibration = load("sensor_calibration.mat");

    N = length(pos);
    GT = pos;

    accel  = (sensor_calibration.Rotw_a * accel')';
    gyro   = (sensor_calibration.Rotw_a * gyro')';
    accel2 = (sensor_calibration.Rotw_a2 * accel2')';

    accel  = accel - sensor_calibration.accel_bias;
    gyro   = gyro - sensor_calibration.gyro_bias;
    accel2 = accel2 - sensor_calibration.accel2_bias;

    fs1 = 104; fs2 = 100; fc = 0.5; order = 2;
    [b1, a1] = butter(order, fc/(fs1/2));
    [b2, a2] = butter(order, fc/(fs2/2));
    accel1_x = filtfilt(b1, a1, accel(:,1));
    accel1_y = filtfilt(b1, a1, accel(:,2));
    accel2_x = filtfilt(b2, a2, accel2(:,1));
    accel2_y = filtfilt(b2, a2, accel2(:,2));

    w1 = 2; w2 = 1;
    fused_accel_x = (w1 * accel1_x + w2 * accel2_x) / (w1 + w2);
    fused_accel_y = (w1 * accel1_y + w2 * accel2_y) / (w1 + w2);
    accel_fused = [fused_accel_x, fused_accel_y, accel(:,3)];
    if plt
        figure()
        plot(fused_accel_y)
        hold on
        plot(fused_accel_x)
        hold off
        title('Smoothed fused Accel x/y');
        legend('y','x');
    end

    %% === Low-pass filter gyro yaw rate ===
    order = 5; cutoff = 4;
    [b, a] = butter(order, cutoff / (104/2));
    gyro_z = filtfilt(b, a, gyro(:,3));
    % Optionally: gyro(:,3) = gyro_z;

    %% === Debug: Plot filtered vs raw gyro yaw rate ===
    if plt
        figure;
        plot(timeVec, gyro(:,3), 'b-', 'DisplayName', 'Raw Gyro Z'); hold on;
        plot(timeVec, gyro_z, 'r-', 'DisplayName', 'Filtered Gyro Z');
        legend();
        xlabel('Time [s]');
        ylabel('Yaw Rate [rad/s]');
        title('Gyro Z-axis (Yaw Rate) - Raw vs Filtered');
        grid on;
    end

    yawGT = quat2eul(rotQuat);
    % Be sure to choose the correct yaw value; here we use the first element.
    yawGT = yawGT(:,1) + pi;
    yawGT = unwrap(yawGT);
    GT = [GT, yawGT];


    %% === EKF Initialization ===
    % Store state as a 5D vector [x, y, vx, vy, yaw].
    X_Est = zeros(N, 5);
    P_Est = cell(N,1);
    % Here we initialize using the first position and the first yaw value.
    X_Est(1,:) = [pos(1,1), pos(1,2), 0, 0, yawGT(1)];
    % Covariance for 5 states
    P_Est{1} = diag([0.01, 0.01, 0.01, 0.01, 0]);
    Q = q;
    % The base R (for the three ToF sensors) is taken from the calibration
    RdiagBase = [sensor_calibration.R_tof_left, sensor_calibration.R_tof_middle, sensor_calibration.R_tof_right];
    scaleFactors = r;
    max_range = 2.0;  % meters
    
    disp(Q)
    
    %% === UPDATE LOOP ===
    for k = 1:N-1
        dt = timeVec(k+1) - timeVec(k);
        if dt <= 0, dt = 1e-2; end
    
        % Prediction
        xk = X_Est(k,:)';
        [x_pred, F] = predictionStepSymbolic(xk, accel_fused(k,:)', gyro_z(k), dt);
        F5 = F(1:5,1:5);
        Q5 = Q(1:5,1:5);
        P_pred = F5 * P_Est{k} * F5' + Q5;
        
        % Measurement prediction
        [z_pred, H] = measurementModel(x_pred);
        z_pred = min(z_pred, max_range);  % clamp overly long predictions

        z_meas = [tof_right(k); tof_front(k); tof_left(k)];
         
    
        % Kalman update
        innovation = z_meas - z_pred;
        R_meas = diag(scaleFactors);
        S = H * P_pred * H' + R_meas;
        K = P_pred * H' / S;
        x_upd = x_pred + K * innovation;
        x_upd(5) = wrapToPi(x_upd(5));
        P_upd = (eye(5) - K * H) * P_pred;
    
        % Store result
        X_Est(k+1,:) = x_upd';
        P_Est{k+1} = P_upd;
    
        % ===== DEBUGGING OUTPUT =====
        if mod(k,150) == 0
            %fprintf('\n--- EKF Step %d ---\n', k);
            %fprintf('Innovation (z_meas - z_pred):\n'); disp(innovation');
            %fprintf('Kalman Gain K (rows for x, y, psi):\n');
            %disp(K([1,2,5], :));  % focus on position and yaw
            %fprintf('Updated yaw estimate: %.3f rad\n', x_upd(5));
            %disp("z_meas (actual ToF readings):");
            %disp(z_meas');
            
            %disp("z_pred (predicted ToF readings):");
            %disp(z_pred');
            %fprintf("Step %d: x=%.2f y=%.2f yaw=%.2f\n", k, x_pred(1), x_pred(2), x_pred(5));


        end
    
        % === Optional: store debug data ===
        % debug_velocity(k,:) = x_upd(3:4);  % uncomment if you want to log velocities
        % debug_psi(k) = x_upd(5);           % store yaw estimate
        % debug_innov(k,:) = innovation';    % store innovations
    
    
       
    
        % Calculate and plot errors
        err_x = X_Est(:,1) - GT(:,1);
        err_y = X_Est(:,2) - GT(:,2);
        err_yaw = wrapToPi(X_Est(:,5) - GT(:,4));
        if plt
            figure;
            subplot(3,1,1); plot(timeVec, err_x); title('Error in x');
            subplot(3,1,2); plot(timeVec, err_y); title('Error in y');
            subplot(3,1,3); plot(timeVec, smooth(err_yaw)); title('Error in yaw');
        end
    end
end

%% === Prediction Step Function (5D) ===
function [x_pred, F] = predictionStepSymbolic(xk, accel, gyroz, dt)
    % This version assumes that the state’s yaw (psi) is defined relative 
    % to the +X axis. (Originally, psi=0 meant facing +Y.)
    %
    % To convert from the old convention (if needed), you could define:
    %   psi_effective = psi - pi/2;
    % Here we assume you have reinitialized so that xk(5) is in the new frame.
    
    % Extract the 5 states: [x, y, vx, vy, psi] where psi=0 => +X direction.
    x   = xk(1);
    y   = xk(2);
    vx  = xk(3);
    vy  = xk(4);
    psi = xk(5);  % New convention: 0 means heading +X.

    gyroz = -gyroz;
    
    % Use the standard rotation matrix for a vector pointing in the direction psi.
    % (In a standard robotics context, if the vehicle’s forward direction is along body x,
    %  then R = [cos(psi) -sin(psi); sin(psi) cos(psi)] maps body-frame into world-frame.)
    R_bw = [cos(psi), -sin(psi); 
            sin(psi),  cos(psi)];
    
    % Accelerometer measurements in the body frame.
    a_x = accel(1);
    a_y = accel(2);
    a_world = R_bw * [a_x; a_y];
    a_wx = a_world(1);
    a_wy = a_world(2);
    
    % Predict new state using a constant-acceleration approximation.
    x_pred = zeros(5,1);
    x_pred(1) = x + vx * dt + 0.5 * a_wx * dt^2;
    x_pred(2) = y + vy * dt + 0.5 * a_wy * dt^2;
    x_pred(3) = vx + a_wx * dt;
    x_pred(4) = vy + a_wy * dt;
    % Update yaw using the current gyro measurement (assumed to be in rad/s)
    x_pred(5) = psi + gyroz * dt;
    
    % Compute the Jacobian F for the 5-state system.
    F = eye(5);
    F(1,3) = dt;
    F(2,4) = dt;
    % For the derivatives of position and velocity with respect to psi,
    % note: d/dpsi{cos(psi)} = -sin(psi) and d/dpsi{sin(psi)} = cos(psi).
    % The chain rule gives:
    F(1,5) = -0.5 * dt^2 * ( a_x * sin(psi) + a_y * cos(psi) );
    F(2,5) =  0.5 * dt^2 * ( a_x * cos(psi) - a_y * sin(psi) );
    F(3,5) = -dt * ( a_x * sin(psi) + a_y * cos(psi) );
    F(4,5) =  dt * ( a_x * cos(psi) - a_y * sin(psi) );
end

function [z_pred, H] = measurementModel(x)
    % Define wall positions (in meters)
    x_front = 1.2;   % Front wall is at x = 1.2 (in front of the vehicle)
    y_left  = 1.2;   % Left wall is at y = 1.2
    y_right = -1.2;  % Right wall is at y = -1.2

    % Extract the necessary state variables
    x_pos = x(1);
    y_pos = x(2);
    psi   = x(5);    % yaw angle measured from +world-x

    % For robustness, define a safe version of cosine (to avoid division by zero)
    epsilon = 1e-4;
    cos_psi = cos(psi);
    safe_cos = sign(cos_psi) * max(abs(cos_psi), epsilon);

    % --- Compute predicted distances for each sensor ---
    % Front Sensor:
    % The sensor ray points in the vehicle heading (psi).
    % Intersection with the wall at x = +1.2:
    t_front = (x_front - x_pos) / safe_cos;

    % Left Sensor:
    % For the sensor oriented at psi+pi/2 the effective cosine is given by
    % sin(psi+pi/2)=cos(psi). Since the left wall is at y = +1.2,
    % the distance is computed as:
    t_left = (y_left - y_pos) / safe_cos;

    % Right Sensor:
    % For the sensor oriented at psi-pi/2, note that sin(psi-pi/2) = -cos(psi).
    % The right wall is at y = -1.2, so:
    t_right = (y_pos - y_right) / safe_cos;
    % Alternatively, if you wish to keep the sign conventions consistent, you 
    % could use an additional negative sign in the denominator (or adjust the sensor 
    % measurement interpretation) but here we assume the sensor reports a positive distance.

    % Assemble the predicted measurement vector:
    % Order: [Front; Right; Left]
    z_pred = [t_front; t_right; t_left];

    % --- Compute the Jacobian H ---
    % H will be a (3x5) matrix. Only states x, y, and psi affect the measurements.
    H = zeros(3, 5);

    % For the Front Sensor:
    %   t_front = (x_front - x_pos) / cos(psi)
    H(1, 1) = -1 / safe_cos;                         % dt_front/dx
    % dt_front/dpsi: differentiate [ (x_front - x_pos)/cos(psi) ]
    H(1, 5) = (x_front - x_pos) * sin(psi) / (safe_cos^2); 
    % Derivatives with respect to y, vx, vy remain zero.

    % For the Right Sensor:
    %   t_right = (y_pos - y_right) / cos(psi)
    H(2, 2) = 1 / safe_cos;                          % dt_right/dy
    % dt_right/dpsi: differentiate [ (y_pos - y_right)/cos(psi) ]
    H(2, 5) = - (y_pos - y_right) * sin(psi) / (safe_cos^2);

    % For the Left Sensor:
    %   t_left = (y_left - y_pos) / cos(psi)
    H(3, 2) = -1 / safe_cos;                         % dt_left/dy
    % dt_left/dpsi:
    H(3, 5) = (y_left - y_pos) * sin(psi) / (safe_cos^2);
    
    % The partial derivatives with respect to the velocity states (columns 3 and 4) are zero.
end
