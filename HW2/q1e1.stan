data {
    int<lower=1> N; // number of observations
    matrix[N, 9] Y; // data matrix
    vector[N] c; // fixed covariate
    vector[N] d; // fixed covariate
    cov_matrix[2] Phi0; // covariance matrix of Inverse Wishart Dist.
}
transformed data {
    vector[2] zero = rep_vector(0, 2);
}
parameters {
    vector[9] u;
    vector[6] lam;
    vector[9] A;
    real b;
    vector[4] gam;
    vector<lower=0.0>[9] sgm2;
    real<lower=0.0> sgd2;
    cov_matrix[2] phx;
    vector[N] eta;
    matrix[N, 2] xi;
}
transformed parameters {
    matrix[N, 9] mu;
    vector[N] nu;
    vector[9] sgm = sqrt(sgm2);
    real sgd = sqrt(sgd2);
    for (i in 1:N){
        nu[i] = b * d[i] + gam[1] * xi[i, 1] + gam[2] * xi[i, 2] + gam[3] * xi[i, 1] * xi[i, 1] + gam[4] * xi[i, 2] * xi[i, 2];        
        mu[i, 1] = u[1] + eta[i];
        mu[i, 2] = u[2] + lam[1] * eta[i];
        mu[i, 3] = u[3] + lam[2] * eta[i];
        mu[i, 4] = u[4] + xi[i, 1];
        mu[i, 5] = u[5] + lam[3] * xi[i, 1];
        mu[i, 6] = u[6] + lam[4] * xi[i, 1];
        mu[i, 7] = u[7] + xi[i, 2];
        mu[i, 8] = u[8] + lam[5] * xi[i, 2];
        mu[i, 9] = u[9] + lam[6] * xi[i, 2];
        for (j in 1:9)
            mu[i, j] = mu[i, j] + A[j] * c[i];
    }
}
model {
    // hyper prior 
    sgm2 ~ inv_gamma(3, 10);
    sgd2 ~ inv_gamma(3, 10);
    
    // prior
    A ~ multi_normal([0.3, 0.5, 0.4, 0.3, 0.5, 0.4, 0.3, 0.5, 0.4], diag_matrix(sgm2));
    u ~ normal(0, 1);
    lam[1] ~ normal(0.9, sgm[2]);
    lam[2] ~ normal(0.7, sgm[3]);
    lam[3] ~ normal(0.9, sgm[5]);
    lam[4] ~ normal(0.7, sgm[6]);
    lam[5] ~ normal(0.9, sgm[8]);
    lam[6] ~ normal(0.7, sgm[9]);
    b ~ normal(0.5, sgd);
    gam[1] ~ normal(0.4, sgd);
    gam[2] ~ normal(0.4, sgd);
    gam[3] ~ normal(0.3, sgd);
    gam[4] ~ normal(0.2, sgd);
    phx ~ inv_wishart(4, Phi0);

    // structural equation
    for (i in 1:N){
        xi[i] ~ multi_normal(zero, phx);
        eta[i] ~ normal(nu[i], sgd);
    }

    // the likelihood
    for (i in 1:N){
        for (j in 1:9){
            Y[i, j] ~ normal(mu[i, j], sgm[j]);
        }
    }
}
