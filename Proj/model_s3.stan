data {
    int<lower=1> N;
    vector[6] Y[N];
    int<lower=0, upper=9> Z[N, 2];
    int<lower=0> age[N];
}
transformed data {
    vector[3] zero = rep_vector(0, 3);
    cov_matrix[3] Phi0 = diag_matrix(rep_vector(1, 3));
}
parameters {
    vector[8] mu;
    vector[4] lambda;
    vector[3] gamma;
    vector<lower=0.0>[8] sgm2;
    real<lower=0.0> sgd2;
    cov_matrix[3] phx;
    vector[N] eta;
    vector[3] xi[N];
    ordered[3] c;
    real c0;
    real b;
}
transformed parameters {
    vector[8] u[N];
    vector[N] nu;
    vector[8] sgm = sqrt(sgm2);
    real sgd = sqrt(sgd2);
    
    for (i in 1:N){
        nu[i] = b * age[i] + gamma[1] * xi[i, 1] + gamma[2] * xi[i, 2] + gamma[3] * xi[i, 3];
        u[i, 1] = mu[1] + eta[i];
        u[i, 2] = mu[2] + xi[i, 1];
        u[i, 3] = mu[3] + lambda[1] * xi[i, 1];
        u[i, 4] = mu[4] + lambda[2] * xi[i, 1];
        u[i, 5] = mu[5] + xi[i, 2];
        u[i, 6] = mu[6] + lambda[3] * xi[i, 2];
        u[i, 7] = mu[7] + xi[i, 3];
        u[i, 8] = mu[8] + lambda[4] * xi[i, 3];
    }
}
model {
    vector[4] theta;
    vector[2] theta0;

    // prior
    mu ~ normal(0, 1);
    lambda[1] ~ normal(0.9, sgm[1]);
    lambda[2] ~ normal(0.7, sgm[2]);
    lambda[3] ~ normal(0.9, sgm[3]);
    lambda[4] ~ normal(0.7, sgm[4]);
    gamma[1] ~ normal(0.1, sgd);
    gamma[2] ~ normal(0.1, sgd);
    gamma[3] ~ normal(0.9, sgd);
    b ~ normal(0.5, sgd);

    sgm2 ~ inv_gamma(9, 4);
    sgd2 ~ inv_gamma(9, 4);

    phx ~ inv_wishart(4, Phi0);

    for (i in 1:N) {
        eta[i] ~ normal(nu[i], sgd);
        xi[i] ~ multi_normal(zero, phx);
    }

    // likelihood
    for (i in 1:N) {
        theta[1] = Phi((c[1] - u[i, 3]) / sgm[3]);
        theta[2] = Phi((c[2] - u[i, 3]) / sgm[3]) - Phi((c[1] - u[i, 3]) / sgm[3]);
        theta[3] = Phi((c[3] - u[i, 3]) / sgm[3]) - Phi((c[2] - u[i, 3]) / sgm[3]);
        theta[4] = 1 - Phi((c[3] - u[i, 3]) / sgm[3]);
        Y[i, 1] ~ normal(u[i, 1], sgm[1]);
        Y[i, 2] ~ normal(u[i, 2], sgm[2]);
        Z[i, 1] - 5 ~ categorical(theta);
        Y[i, 3] ~ normal(u[i, 4], sgm[4]);
        Y[i, 4] ~ normal(u[i, 5], sgm[5]);
        Y[i, 5] ~ normal(u[i, 6], sgm[6]);
        theta0[1] = Phi((c0 - u[i, 7]) / sgm[7]);
        theta0[2] = 1 - theta0[1];
        Z[i, 2] + 1 ~ categorical(theta0);
        Y[i, 6] ~ normal(u[i, 8], sgm[8]);
    }
}