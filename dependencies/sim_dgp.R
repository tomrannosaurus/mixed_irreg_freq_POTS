simulate_mixedfreq_data <- function(
    
  # Basic parameters
  p = 3,
  freq_min = 0.01,
  freq_max = 5,
  regularity = 0.5,
  
  # "Sync groups" control:
  num_sync_groups = 1,
  
  # Process controls
  process_types = c("ar1", "rw", "seasonal"), 
  ar1_phi = 0.8,
  ar1_sigma = 1.0,
  rw_sigma = 1.0,
  seasonal_period = 24,
  seasonal_amplitude = 1,
  seasonal_noise_sd = 0.2,
  
  # random-drift parameters
  seasonal_drift_min = 0,     # lower bound for random drift
  seasonal_drift_max = 0.05,  # upper bound for random drift
  
  extra_sine_amp = 0.3,       # amplitude of the extra wave
  extra_sine_period = 12,     # period of the extra wave
  
  # Missingness
  missing_prop = 0,
  block_missing = FALSE,
  block_length = 1,
  
  # Outcome: single measurement
  outcome_time = 10,
  beta = c(0, 1, -0.5, 0.5),
  
  # Complex outcome options
  use_lags = FALSE,
  lag_hours = 1,
  integrate_features = FALSE,
  
  # RNG seed
  seed = 123
) {
  # ---------------------------------------------------------------------------
  # This function simulates data for ONE individual with p features.
  # The user can run it multiple times for multiple individuals.
  #
  # 1) "Sync groups": we partition p variables into 'num_sync_groups' groups.
  #    Each group has a single set of sampling times, but each variable in that
  #    group has its own process realization (AR(1), RW, etc.).
  # 2) We then optionally introduce missingness (random or block).
  # 3) We generate exactly ONE outcome Y at time = outcome_time.
  #    Y ~ Bernoulli( logit( beta0 + sum_j beta_j * X_j(t) ) ).
  #    The X_j(t) can be:
  #       - X_j(t) if integrate_features=FALSE & use_lags=FALSE
  #       - X_j(t - lag_hours) if use_lags=TRUE
  #       - Integral( X_j over [0, t] ) if integrate_features=TRUE
  #
  # Returned:
  #   - $variables_raw: list of p data.frames
  #       each data.frame has columns (time, value, variable)
  #   - $outcome_df: data.frame with a single row => (time = outcome_time, Y)
  #
  # ---------------------------------------------------------------------------
  
  set.seed(seed)
  
  # Validate lengths
  if (length(beta) != (p + 1)) {
    stop("`beta` must have length p+1 (intercept + p coefficients).")
  }
  
  # If process_types is length 1, replicate it for each variable
  if (length(process_types) == 1) {
    process_types <- rep(process_types, p)
  } else if (length(process_types) != p) {
    stop("`process_types` must be length 1 or length p.")
  }
  
  if (num_sync_groups < 1 || num_sync_groups > p) {
    stop("`num_sync_groups` must be between 1 and p.")
  }
  
  # Assign each variable to a "sync group"
  group_assign <- sample.int(num_sync_groups, size = p, replace = TRUE)
  
  # Generate sampling times for each group
  generate_times_for_group <- function() {
    freq_j <- runif(1, freq_min, freq_max)
    approx_n_j <- ceiling(freq_j * 10)  # horizon is [0,10]
    
    if (regularity == 1) {
      times_j <- seq(0, 10, length.out = approx_n_j)
    } else if (regularity == 0) {
      times_j <- sort(runif(approx_n_j, 0, 10))
    } else {
      n_reg <- floor(regularity * approx_n_j)
      n_rnd <- approx_n_j - n_reg
      times_reg <- if (n_reg > 1) {
        seq(0, 10, length.out = n_reg)
      } else if (n_reg == 1) {
        runif(n_reg, 0, 10)
      } else {
        numeric(0)
      }
      times_rnd <- if (n_rnd > 0) {
        runif(n_rnd, 0, 10)
      } else {
        numeric(0)
      }
      times_j <- sort(c(times_reg, times_rnd))
    }
    times_j
  }
  
  group_times <- vector("list", num_sync_groups)
  for (g in seq_len(num_sync_groups)) {
    group_times[[g]] <- generate_times_for_group()
  }
  
  gen_process_values <- function(proc_type, times_j, drift_j) {
    n_j <- length(times_j)
    if (n_j == 0) return(numeric(0))
    
    x_vals <- numeric(n_j)
    
    if (proc_type == "ar1") {
      # AR(1)
      x_vals[1] <- rnorm(1, 0, ar1_sigma)
      for (k in 2:n_j) {
        x_vals[k] <- ar1_phi * x_vals[k-1] + rnorm(1, 0, ar1_sigma * sqrt(1 - ar1_phi^2))
      }
      
    } else if (proc_type == "rw") {
      # random walk
      x_vals[1] <- rnorm(1, 0, rw_sigma)
      for (k in 2:n_j) {
        x_vals[k] <- x_vals[k-1] + rnorm(1, 0, rw_sigma)
      }
      
    } else if (proc_type == "seasonal") {
      # seasonal + random drift + extra sine wave
      for (k in seq_len(n_j)) {
        t_now <- times_j[k]
        
        # Start with two sine waves
        mean_val <- seasonal_amplitude * sin(2 * pi * t_now / seasonal_period) +
          0.5 * seasonal_amplitude * sin(2 * pi * t_now / (seasonal_period / 2))
        
        # Add random drift
        mean_val <- mean_val + drift_j * t_now
        
        # Add a third sine wave
        mean_val <- mean_val + extra_sine_amp * seasonal_amplitude *
          sin(2 * pi * t_now / extra_sine_period + phase_offset_j)
        
        # Observed value
        x_vals[k] <- rnorm(1, mean_val, seasonal_noise_sd)
      }
      
    } else {
      # fallback: standard normal
      x_vals <- rnorm(n_j, 0, 1)
    }
    
    x_vals
  }
  
  var_list <- vector("list", p)
  
  for (j in seq_len(p)) {
    g_j <- group_assign[j]
    times_j <- group_times[[g_j]]
    ptype_j <- process_types[j]
    
    # Draw a random drift if it's seasonal; otherwise 0
    if (ptype_j == "seasonal") {
      drift_j <- runif(1, seasonal_drift_min, seasonal_drift_max)
      phase_offset_j <- runif(1, 0, 2*pi)  # random offset
    } else {
      drift_j <- 0
      phase_offset_j <- 0
    }
    
    # generate the underlying process
    x_vals <- gen_process_values(ptype_j, times_j, drift_j)
    
    # build the data frame
    df_j <- data.frame(
      time = times_j,
      value = x_vals,
      variable = paste0("X", j)
    )
    
    # missingness
    if (missing_prop > 0 && nrow(df_j) > 0) {
      keep_idx <- sample.int(nrow(df_j), size = floor((1 - missing_prop) * nrow(df_j)))
      df_j <- df_j[keep_idx, , drop = FALSE]
    }
    if (block_missing && nrow(df_j) > 1) {
      block_center <- runif(1, 0, 10)
      block_start <- block_center - block_length / 2
      block_end   <- block_center + block_length / 2
      df_j <- df_j[!(df_j$time >= block_start & df_j$time <= block_end), ]
    }
    
    df_j <- df_j[order(df_j$time), ]
    var_list[[j]] <- df_j
  }
  
  # Generate a SINGLE outcome Y at time = outcome_time
  if (outcome_time < 0) {
    stop("`outcome_time` must be >= 0 (assuming timeline [0,t]).")
  }
  
  get_feature_value <- function(df_j, t_out) {
    if (nrow(df_j) == 0) return(NA_real_)
    approx_j <- approx(df_j$time, df_j$value, xout = t_out, rule = 2)
    approx_j$y
  }
  
  get_feature_integral <- function(df_j, t_out) {
    if (nrow(df_j) == 0) return(0)
    ord <- order(df_j$time)
    x_time <- df_j$time[ord]
    x_val  <- df_j$value[ord]
    if (t_out < 0) return(0)
    if (x_time[1] > 0) {
      x_time <- c(0, x_time)
      x_val  <- c(0, x_val)
    }
    dt <- diff(x_time)
    midvals <- (x_val[-1] + x_val[-length(x_val)]) / 2
    trap <- dt * midvals
    csum <- c(0, cumsum(trap))
    approx_res <- approx(x_time, csum, xout = t_out, rule = 2)
    approx_res$y
  }
  
  X_out <- numeric(p)
  
  for (j in seq_len(p)) {
    df_j <- var_list[[j]]
    
    # if variable has <2 valid points, skip interpolation
    if (nrow(df_j) < 2 || sum(!is.na(df_j$value)) < 2) {
      X_out[j] <- 0
      next
    }
    
    if (!use_lags && !integrate_features) {
      val_j <- get_feature_value(df_j, outcome_time)
    } else if (use_lags) {
      t_query <- outcome_time - lag_hours
      if (t_query < 0) t_query <- 0
      val_j <- get_feature_value(df_j, t_query)
    } else if (integrate_features) {
      val_j <- get_feature_integral(df_j, outcome_time)
    } else {
      val_j <- NA_real_
    }
    
    X_out[j] <- ifelse(is.na(val_j), 0, val_j)
  }
  
  # logistic outcome
  linpred <- beta[1] + sum(X_out * beta[-1])
  prob    <- if (runif(1) < 0.5) pnorm(linpred) else 1 - exp(-exp(linpred))
  Y <- rbinom(1, size = 1, prob = prob)
  
  outcome_df <- data.frame(
    time = outcome_time,
    Y = Y,
    true_prob = prob
  )
  
  list(
    variables_raw = var_list,
    outcome_df    = outcome_df
  )
}



simulate_experiment_dataset <- function(
    
  # Experiment-wide parameters
  N = 100,
  p = 3,
  freq_min = 0.5,
  freq_max = 2,
  regularity = 0.3,
  
  # Sync group and process parameters
  num_sync_groups = 1,
  process_types = "ar1",   # can be a single string or a vector of length p
  ar1_phi = 0.8,
  ar1_sigma = 1.0,
  rw_sigma = 1.0,
  
  # Seasonal parameters (including random drift and extra wave)
  seasonal_period = 24,
  seasonal_amplitude = 1,
  seasonal_noise_sd = 0.2,
  seasonal_drift_min = 0,     # lower bound for random drift
  seasonal_drift_max = 0.05,  # upper bound for random drift
  extra_sine_amp = 0.3,       # amplitude of the extra wave
  extra_sine_period = 12,     # period of the extra wave
  
  # Missingness parameters
  missing_prop = 0,
  block_missing = FALSE,
  block_length = 1,
  
  # Outcome parameters
  outcome_time = 10,
  beta = c(0, 1, -0.5),   # must have length p+1 if p=2, or adapt accordingly
  use_lags = FALSE,
  lag_hours = 1,
  integrate_features = FALSE,
  
  # RNG seed for reproducibility
  master_seed = 123
) {
  # ---------------------------------------------------------------------------
  # This function simulates a dataset of N individuals using the single-subject
  # DGP 'simulate_mixedfreq_data()', which you've modified to include:
  #  - random drift (seasonal_drift_min, seasonal_drift_max)
  #  - multiple sine waves (extra_sine_amp, extra_sine_period)
  #
  # Args:
  #   N: number of subjects
  #   p: number of features
  #   freq_min, freq_max, regularity: sampling times controls
  #   num_sync_groups: how many groups share sampling times
  #   process_types: "ar1", "rw", "seasonal", etc. (length 1 or p)
  #   ar1_phi, ar1_sigma, rw_sigma, etc.: parameters for the underlying processes
  #   seasonal_*: parameters for the extended seasonal model (noise, drift, extra wave)
  #   missing_prop, block_missing, block_length: missingness parameters
  #   outcome_time: single time at which Y is measured for each individual
  #   beta: logistic coefficients (length p+1)
  #   use_lags, lag_hours, integrate_features: how outcome depends on X
  #   master_seed: overall seed
  #
  # Returns:
  #   A list with two data frames:
  #   - $vars_long   : a long-format table of all variable measurements
  #       columns: (ID, time, variable, value)
  #   - $outcome_df  : a data frame of outcomes, with columns: (ID, time, Y)
  # ---------------------------------------------------------------------------
  
  # set a master seed for replicability
  set.seed(master_seed)
  
  # We'll store the time-series measurements in a list, outcomes in a separate DF
  all_vars_long <- vector("list", N)
  all_outcomes  <- data.frame()
  
  # Loop over N individuals
  for (i in seq_len(N)) {
    
    # Generate data for subject i using your updated 'simulate_mixedfreq_data()'
    sim_i <- simulate_mixedfreq_data(
      p                   = p,
      freq_min            = freq_min,
      freq_max            = freq_max,
      regularity          = regularity,
      num_sync_groups     = num_sync_groups,
      process_types       = process_types,
      ar1_phi             = ar1_phi,
      ar1_sigma           = ar1_sigma,
      rw_sigma            = rw_sigma,
      
      # Pass all the seasonal arguments down:
      seasonal_period     = seasonal_period,
      seasonal_amplitude  = seasonal_amplitude,
      seasonal_noise_sd   = seasonal_noise_sd,
      seasonal_drift_min  = seasonal_drift_min,
      seasonal_drift_max  = seasonal_drift_max,
      extra_sine_amp      = extra_sine_amp,
      extra_sine_period   = extra_sine_period,
      
      missing_prop        = missing_prop,
      block_missing       = block_missing,
      block_length        = block_length,
      
      outcome_time        = outcome_time,
      beta                = beta,
      use_lags            = use_lags,
      lag_hours           = lag_hours,
      integrate_features  = integrate_features,
      
      # For each subject, randomize the seed further:
      seed = sample.int(.Machine$integer.max, 1)  
    )
    
    # sim_i$variables_raw is a list of data frames (one per feature).
    # combine them into a single 'long' data frame for subject i.
    subject_vars <- do.call(rbind, sim_i$variables_raw)
    # Add the subject ID
    subject_vars$ID <- i
    # Reorder columns for clarity
    subject_vars <- subject_vars[, c("ID", "time", "variable", "value")]
    
    # Also store the outcome (just 1 row: time=outcome_time, Y=binary)
    out_df <- data.frame(
      ID = i,
      time = sim_i$outcome_df$time,
      Y = sim_i$outcome_df$Y,
      true_prob = sim_i$outcome_df$true_prob
    )
    
    # Accumulate
    all_vars_long[[i]] <- subject_vars
    all_outcomes <- rbind(all_outcomes, out_df)
  }
  
  # Combine all variable-measurement data frames
  vars_long_df <- do.call(rbind, all_vars_long)
  
  # Return as a list
  list(
    vars_long  = vars_long_df,
    outcome_df = all_outcomes
  )
}