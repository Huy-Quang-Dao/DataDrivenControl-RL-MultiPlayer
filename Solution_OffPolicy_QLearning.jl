# Code for paper: Output Feedback H1 Control for Linear
# Discrete-Time Multi-Player Systems With
# Multi-Source Disturbances Using Off-Policy
# Q-Learning.

# Programing Language : Julia 
# Method : Off-Policy
# Purpose : Practice and Research

# Lib
using LinearAlgebra
using Plots
using Kronecker
# Model
A_c = [0.4126 0 -0.0196;
0.0333 0.5207 -0.0413;
-0.0101 0 0.2571];
B1_c=[-1.7734,0.0928,-0.0424][:,:];
B2_c=[0.0696,0.4658,-0.0930][:,:];
B3_c=[0.0734,0.1051,2.0752][:,:];
E1_c=[0.123956,-0.37895,0.12698][:,:];
E2_c=[-0.08564,0.1268,0.036984][:,:];
C=[1 0 -1;-2 1 0;1 1 1];
A = C*A_c*pinv(C'*C)*C';
B1 = C*B1_c;
B2 = C*B2_c;
B3 = C*B3_c;
E1 = C*E1_c;
E2 = C*E2_c;
# Value function parameters
Q = Matrix(Diagonal([10,10,10]));
R1=[10][:,:];
R2=R1;R3=R1;
γ=5;

# Nash equilibrium
K1_s = -[0.0295 -0.0829 0.0209];
K2_s = -[-0.1904 -0.1612 -0.1622];
K3_s = -[0.0253 -0.0240 -0.0756];
Kd1_s = -[-0.0599 -0.0546 -0.0503];
Kd2_s = -[0.0183 0.0186 0.0169];

# 
n = size(A, 2);
m1 = size(B1, 2);
m2 = size(B2, 2);
m3 = size(B3, 2);
p1 = size(E1, 2);
p2 = size(E2, 2);
add =20;
f = n^2 + m1^2 + m2^2 + m3^2 + p1^2 + p2^2 + m1 * m2 + m1 * m3 + m1 * p1 + m1*p2 + m2 * m3 + m2 * p1 + m2*p2 + m3 * p1 + m3*p2 + p1*p2 + n * (m1 + m2 + m3 + p1+p2) + add;
n_learn = 20;
x0 = [10, -10, 10] # Initial state
K1_0 = [0.1 0 -0];
K2_0 = [-0.6 0 0];
K3_0 = [0.1 -0.0 0];
Kd1_0 = [-0.5 0 0];
Kd2_0 = [-0.5 0 0];
global i = 1;
K1 = [K1_0];
K2 = [K2_0];
K3 = [K3_0];
Kd1= [Kd1_0];
Kd2= [Kd2_0];

# phi1 = []; phi2 = []; phi3 = []; phi4 = []; phi5 = []; phi6 = []; phi7 = [];
# phi8 = []; phi9 = []; phi10 = []; phi11 = []; phi12 = []; phi13 = []; phi14 = []; phi15 = [];
# phi16 = []; phi17 = []; phi18 = []; phi19 = []; phi20 = []; phi21 = [];
# phi = []; psi = [];

# Collect data to use Off-policy RL algorithm
x = zeros(n, f+1);
x[:, 1]=[0.2;-0.65;1.55];
u1 = zeros(m1, f);
u2 = zeros(m2, f);
u3 = zeros(m3, f);
w1 = zeros(p1, f);
w2 = zeros(p2, f);

for k in 1:f
    e1 = 0.1 * sin(k);
    e2 = e2 = 0.1 * sin(0.1 * k);
    e3 = e3 = 0.1 * sin(0.3 * k);
    # probing noise
    u1[:,k] = -K1[i] * x[:, k] + [e1];
    u2[:,k] = -K2[i] * x[:, k] + [e2];
    u3[:,k] = -K3[i] * x[:, k] + [e3];
    w1[:, k] = [exp(-0.01*k) * sin(0.1*k)];
    w2[:, k] = [exp(-0.1*k) * sin(2.1*k)];
    x[:, k+1] = A * x[:, k] + B1 * u1[:, k] + B2 * u2[:, k] + B3 * u3[:, k] + E1 * w1[:, k]+ E2 * w2[:, k];
end

while true 
    global Phi = zeros(Float64, f, f-add);
    global Psi = zeros(Float64, f, 1);
    for k in 1:f
        global phi1 = kron(x[:, k]', x[:, k]') - kron(x[:, k+1]', x[:, k+1]');
        global phi2 = 2 * kron((u1[:, k] + K1[i] * x[:, k])', x[:, k]');
        global phi3 = 2 * kron((u2[:, k] + K2[i] * x[:, k])', x[:, k]');
        global phi4 = 2 * kron((u3[:, k] + K3[i] * x[:, k])', x[:, k]');
        global phi5 = 2 * kron((w1[:, k] + Kd1[i] * x[:, k])', x[:, k]');
        global phi6 = 2 * kron((w2[:, k] + Kd2[i] * x[:, k])', x[:, k]');
        global phi7 = kron((u1[:, k] + K1[i] * x[:, k])', (u1[:, k] - K1[i] * x[:, k])');
        global phi8 = 2 * (kron(u2[:, k]', u1[:, k]') - kron((-K2[i] * x[:, k])', (-K1[i] * x[:, k])'));
        global phi9 = 2 * (kron(u3[:, k]', u1[:, k]') - kron((-K3[i] * x[:, k])', (-K1[i] * x[:, k])'));
        global phi10 = 2 * (kron(w1[:, k]', u1[:, k]') - kron((-Kd1[i] * x[:, k])', (-K1[i] * x[:, k])'));
        global phi11 = 2 * (kron(w2[:, k]', u1[:, k]') - kron((-Kd2[i] * x[:, k])', (-K1[i] * x[:, k])'));
        global phi12 = kron((u2[:, k] + K2[i] * x[:, k])', (u2[:, k] - K2[i] * x[:, k])');
        global phi13 = 2 * (kron(u3[:, k]', u2[:, k]') - kron((-K3[i] * x[:, k])', (-K2[i] * x[:, k])'));
        global phi14 = 2 * (kron(w1[:, k]', u2[:, k]') - kron((-Kd1[i] * x[:, k])', (-K2[i] * x[:, k])'));
        global phi15 = 2 * (kron(w2[:, k]', u2[:, k]') - kron((-Kd2[i] * x[:, k])', (-K2[i] * x[:, k])'));
        global phi16 = kron((u3[:, k] + K3[i] * x[:, k])', (u3[:, k] - K3[i] * x[:, k])');
        global phi17 = 2 * (kron(w1[:, k]', u3[:, k]') - kron((-Kd1[i] * x[:, k])', (-K3[i] * x[:, k])'));
        global phi18 = 2 * (kron(w2[:, k]', u3[:, k]') - kron((-Kd2[i] * x[:, k])', (-K3[i] * x[:, k])'));
        global phi19 = kron((w1[:, k] + Kd1[i] * x[:, k])', (w1[:, k] - Kd1[i] * x[:, k])');
        global phi20 = 2 * (kron(w2[:, k]', w1[:, k]') - kron((-Kd2[i] * x[:, k])', (-Kd1[i] * x[:, k])'));
        global phi21 = kron((w2[:, k] + Kd2[i] * x[:, k])', (w2[:, k] - Kd2[i] * x[:, k])');
        global phi =hcat(phi1, phi2, phi3, phi4, phi5, phi6, phi7, phi8, phi9, phi10, phi11, phi12, phi13, phi14, phi15,phi16, phi17, phi18, phi19, phi20, phi21);
        global psi = (x[:, k]') * (Q + K1[i]' * R1 * K1[i] + K2[i]' * R2 * K2[i] + K3[i]' * R3 * K3[i] - γ^2 * (Kd1[i]' * Kd1[i]+Kd2[i]' * Kd2[i])) * x[:, k];
        # append!(Phi, phi);
        # append!(Psi, psi);
        Phi[k,:] = phi;
        Psi[k] = psi;
    end
    global vec_X = pinv(Phi' * Phi) * Phi' * Psi;
    X1 = vec_X[1:n^2];
    X2 = vec_X[n^2+1:n^2+n*m1];
    X3 = vec_X[n^2+n*m1+1:n^2+n*m1+n*m2];
    X4 = vec_X[n^2+n*m1+n*m2+1:n^2+n*m1+n*m2+n*m3];
    X5 = vec_X[n^2+n*m1+n*m2+n*m3+1:n^2+n*m1+n*m2+n*m3+n*p1];
    X6 = vec_X[n^2+n*m1+n*m2+n*m3+n*p1+1:n^2+n*m1+n*m2+n*m3+n*p1+n*p2];
    X7 = vec_X[n^2+n*m1+n*m2+n*m3+n*p1+n*p2+1:n^2+n*m1+n*m2+n*m3+n*p1+n*p2+m1^2];
    X8 = vec_X[n^2+n*m1+n*m2+n*m3+n*p1+n*p2+m1^2+1:n^2+n*m1+n*m2+n*m3+n*p1+n*p2+m1^2+m1*m2];
    X9 = vec_X[n^2+n*m1+n*m2+n*m3+n*p1+n*p2+m1^2+m1*m2+1:n^2+n*m1+n*m2+n*m3+n*p1+n*p2+m1^2+m1*m2+m1*m3];
    X10 = vec_X[n^2+n*m1+n*m2+n*m3+n*p1+n*p2+m1^2+m1*m2+m1*m3+1:n^2+n*m1+n*m2+n*m3+n*p1+n*p2+m1^2+m1*m2+m1*m3+m1*p1];
    X11 = vec_X[n^2+n*m1+n*m2+n*m3+n*p1+n*p2+m1^2+m1*m2+m1*m3+m1*p1+1:n^2+n*m1+n*m2+n*m3+n*p1+n*p2+m1^2+m1*m2+m1*m3+m1*p1+m1*p2];
    X12 = vec_X[n^2+n*m1+n*m2+n*m3+n*p1+n*p2+m1^2+m1*m2+m1*m3+m1*p1+m1*p2+1:n^2+n*m1+n*m2+n*m3+n*p1+n*p2+m1^2+m1*m2+m1*m3+m1*p1+m1*p2+m2^2];
    X13 = vec_X[n^2+n*m1+n*m2+n*m3+n*p1+n*p2+m1^2+m1*m2+m1*m3+m1*p1+m1*p2+m2^2+1:n^2+n*m1+n*m2+n*m3+n*p1+n*p2+m1^2+m1*m2+m1*m3+m1*p1+m1*p2+m2^2+m2*m3];
    X14 = vec_X[n^2+n*m1+n*m2+n*m3+n*p1+n*p2+m1^2+m1*m2+m1*m3+m1*p1+m1*p2+m2^2+m2*m3+1:n^2+n*m1+n*m2+n*m3+n*p1+n*p2+m1^2+m1*m2+m1*m3+m1*p1+m1*p2+m2^2+m2*m3+m2*p1];
    X15 = vec_X[n^2+n*m1+n*m2+n*m3+n*p1+n*p2+m1^2+m1*m2+m1*m3+m1*p1+m1*p2+m2^2+m2*m3+m2*p1+1:n^2+n*m1+n*m2+n*m3+n*p1+n*p2+m1^2+m1*m2+m1*m3+m1*p1+m1*p2+m2^2+m2*m3+m2*p1+m2*p2];
    X16 = vec_X[n^2+n*m1+n*m2+n*m3+n*p1+n*p2+m1^2+m1*m2+m1*m3+m1*p1+m1*p2+m2^2+m2*m3+m2*p1+m2*p2+1:n^2+n*m1+n*m2+n*m3+n*p1+n*p2+m1^2+m1*m2+m1*m3+m1*p1+m1*p2+m2^2+m2*m3+m2*p1+m2*p2+m3^2];
    X17 = vec_X[n^2+n*m1+n*m2+n*m3+n*p1+n*p2+m1^2+m1*m2+m1*m3+m1*p1+m1*p2+m2^2+m2*m3+m2*p1+m2*p2+m3^2+1:n^2+n*m1+n*m2+n*m3+n*p1+n*p2+m1^2+m1*m2+m1*m3+m1*p1+m1*p2+m2^2+m2*m3+m2*p1+m2*p2+m3^2+m3*p1];
    X18 = vec_X[n^2+n*m1+n*m2+n*m3+n*p1+n*p2+m1^2+m1*m2+m1*m3+m1*p1+m1*p2+m2^2+m2*m3+m2*p1+m2*p2+m3^2+m3*p1+1:n^2+n*m1+n*m2+n*m3+n*p1+n*p2+m1^2+m1*m2+m1*m3+m1*p1+m1*p2+m2^2+m2*m3+m2*p1+m2*p2+m3^2+m3*p1+m3*p2];
    X19 = vec_X[n^2+n*m1+n*m2+n*m3+n*p1+n*p2+m1^2+m1*m2+m1*m3+m1*p1+m1*p2+m2^2+m2*m3+m2*p1+m2*p2+m3^2+m3*p1+m3*p2+1:n^2+n*m1+n*m2+n*m3+n*p1+n*p2+m1^2+m1*m2+m1*m3+m1*p1+m1*p2+m2^2+m2*m3+m2*p1+m2*p2+m3^2+m3*p1+m3*p2+p1^2];
    X20 = vec_X[n^2+n*m1+n*m2+n*m3+n*p1+n*p2+m1^2+m1*m2+m1*m3+m1*p1+m1*p2+m2^2+m2*m3+m2*p1+m2*p2+m3^2+m3*p1+m3*p2+p1^2+1:n^2+n*m1+n*m2+n*m3+n*p1+n*p2+m1^2+m1*m2+m1*m3+m1*p1+m1*p2+m2^2+m2*m3+m2*p1+m2*p2+m3^2+m3*p1+m3*p2+p1^2+p1*p2];
    X21 = vec_X[n^2+n*m1+n*m2+n*m3+n*p1+n*p2+m1^2+m1*m2+m1*m3+m1*p1+m1*p2+m2^2+m2*m3+m2*p1+m2*p2+m3^2+m3*p1+m3*p2+p1^2+p1*p2+1:end];

    X1 = reshape(X1,(n,n));
    X2 = reshape(X2,(n,m1));
    X3 = reshape(X3,(n,m2));
    X4 = reshape(X4,(n,m3));
    X5 = reshape(X5,(n,p1));
    X6 = reshape(X6,(n,p2));
    X7 = reshape(X7,(m1,m1));
    X8 = reshape(X8,(m1,m2));
    X9 = reshape(X9,(m1,m3));
    X10 = reshape(X10,(m1,p1));
    X11 = reshape(X11,(m1,p2));
    X12 = reshape(X12,(m2,m2));
    X13 = reshape(X13,(m2,m3));
    X14 = reshape(X14,(m2,p1));
    X15 = reshape(X15,(m2,p2));
    X16 = reshape(X16,(m3,m3));
    X17 = reshape(X17,(m3,p1));
    X18 = reshape(X18,(m3,p2));
    X19 = reshape(X19,(p1,p1));
    X20 = reshape(X20,(p1,p2));
    X21 = reshape(X21,(p2,p2));

    K1_u = pinv(R1+X7) * (X2' - (X8 *K2[i] + X9 * K3[i] + X10 * Kd1[i] + X11 * Kd2[i]));
    K2_u = pinv(R2+X12) * (X3' - (X8' * K1[i] + X13 * K3[i] + X14 * Kd1[i] + X15 * Kd2[i]));
    K3_u = pinv(R3+X16) * (X4' - (X9' * K1[i] + X13' * K2[i] + X17 * Kd1[i] + X18 * Kd2[i]));
    Kd1_u = pinv(X19 -γ^2*I(p1)) * (X5' - (X10' * K1[i] + X14' * K2[i] + X17' * K3[i] + X20*Kd2[i]));
    Kd2_u = pinv(X21-γ^2*I(p2)) * (X6' - (X11' * K1[i] + X15' * K2[i] + X18' * K3[i] + X20'*Kd1[i]));
    # Find Optimal Solution Step by Step
    push!(K1, K1_u);
    push!(K2, K2_u);
    push!(K3, K3_u);
    push!(Kd1, Kd1_u);
    push!(Kd2, Kd2_u);

    global i = i+1;
    if i>n_learn
        break
    end
end

# Calculate Error
dK1 = zeros(n_learn, 1);
dK2 = zeros(n_learn, 1);
dK3 = zeros(n_learn, 1);
dKd1 = zeros(n_learn, 1);
dKd2 = zeros(n_learn, 1);

for j in 1:n_learn
    dK1[j] = norm(K1[j] - K1_s);
    dK2[j] = norm(K2[j] - K2_s);
    dK3[j] = norm(K3[j] - K3_s);
    dKd1[j] = norm(Kd1[j] - Kd1_s);
    dKd2[j] = norm(Kd2[j] - Kd2_s);
end