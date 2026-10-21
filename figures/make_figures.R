# figures/make_figures.R
#
# Builds every figure and summary table of the CAPRI study from the CSVs in
# experiments/results/ (written by experiments/run_all.py and the forensic
# runners). Nothing here is typed in by hand: each number comes from a CSV,
# and each figure lists its source CSV in its caption line.
#
# Run from the repository root:
#   Rscript figures/make_figures.R
# Output: figures/*.pdf, figures/*.png, figures/tables/*.csv
#
# Requires: ggplot2, dplyr, tidyr, patchwork, readr, scales.

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(patchwork)
  library(readr)
  library(scales)
})

args_all <- commandArgs(trailingOnly = FALSE)
script_arg <- sub("^--file=", "", args_all[grep("^--file=", args_all)])
ROOT <- if (length(script_arg)) normalizePath(file.path(dirname(script_arg), "..")) else getwd()
RES <- file.path(ROOT, "experiments", "results")
OUT <- file.path(ROOT, "figures")
TAB <- file.path(OUT, "tables")
dir.create(TAB, recursive = TRUE, showWarnings = FALSE)

# CSVs carry a provenance header of '#' lines. Logical columns are written by
# pandas as True/False; read everything as character first, then convert.
read_result <- function(name) {
  path <- file.path(RES, name)
  if (!file.exists(path)) stop("missing result file: ", path, " (run experiments/run_all.py first)")
  df <- read_csv(path, comment = "#", col_types = cols(.default = col_character()), progress = FALSE)
  df[] <- lapply(df, function(col) {
    low <- tolower(col)
    if (all(low %in% c("true", "false", NA, ""))) return(low == "true")
    num <- suppressWarnings(as.numeric(col))
    if (all(is.na(num) == (is.na(col) | col == ""))) return(num)
    col
  })
  df
}

source_note <- function(...) paste0("Source: ", paste(c(...), collapse = ", "))

theme_capri <- function() {
  theme_minimal(base_size = 11) +
    theme(panel.grid.minor = element_blank(), legend.position = "bottom",
          plot.title = element_text(face = "bold"), plot.caption = element_text(colour = "grey40", size = 8))
}

save_fig <- function(plot, name, width = 7, height = 4.5) {
  ggsave(file.path(OUT, paste0(name, ".pdf")), plot, width = width, height = height)
  ggsave(file.path(OUT, paste0(name, ".png")), plot, width = width, height = height, dpi = 200, bg = "white")
  cat("  wrote figures/", name, ".{pdf,png}\n", sep = "")
}

save_table <- function(df, name) {
  write_csv(df, file.path(TAB, paste0(name, ".csv")))
  cat("  wrote figures/tables/", name, ".csv\n", sep = "")
}

wilson <- function(k, n, z = 1.96) {
  p <- k / n
  centre <- (p + z^2 / (2 * n)) / (1 + z^2 / n)
  half <- z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2)) / (1 + z^2 / n)
  list(lo = pmax(0, centre - half), hi = pmin(1, centre + half))
}

cat("CAPRI figures and tables from", RES, "\n")

# ---------------------------------------------------------------------------
# Projective invariance
# ---------------------------------------------------------------------------
exp2 <- read_result("invariance.csv")
save_table(exp2 %>% group_by(Stratum) %>%
             summarise(N = n(), N_J_Defined = sum(is.finite(Max_J_RelError)),
                       Median_Max_J_RelError = median(Max_J_RelError, na.rm = TRUE),
                       Max_Max_J_RelError = if (any(is.finite(Max_J_RelError))) max(Max_J_RelError, na.rm = TRUE) else NA_real_,
                       Median_Rho_RelError = median(Rho_RelError, na.rm = TRUE), .groups = "drop"),
           "invariance")

# ---------------------------------------------------------------------------
# Stratification: threshold vs rank classifier
# ---------------------------------------------------------------------------
exp3 <- read_result("stratification_classifiers.csv")
t3 <- exp3 %>% group_by(Noise_Sigma) %>%
  summarise(N = n(), Threshold_Accuracy = mean(Threshold_Classifier == True_Stratum),
            Rank_Accuracy = mean(Rank_Classifier == True_Stratum), .groups = "drop")
save_table(t3, "stratification_accuracy")
save_table(exp3 %>% count(Noise_Sigma, True_Stratum, Rank_Classifier, name = "N"), "stratification_confusion")

# ---------------------------------------------------------------------------
# Conditioning: L(S) ROC and greedy coverage
# ---------------------------------------------------------------------------
roc <- read_result("conditioning_roc.csv") %>% arrange(FPR, TPR)
auc <- sum(diff(roc$FPR) * (head(roc$TPR, -1) + tail(roc$TPR, -1)) / 2)
p2a <- ggplot(roc, aes(FPR, TPR)) + geom_line(colour = "#1f77b4", linewidth = 0.9) +
  geom_abline(linetype = "dashed", colour = "grey60") + coord_equal() +
  labs(title = "(a) ROC of L(S)", subtitle = sprintf("generic vs degenerate tuples; AUC = %.3f", auc),
       x = "False positive rate", y = "True positive rate") + theme_capri()
cov <- read_result("conditioning_greedy_vs_random.csv") %>% group_by(Budget_k) %>%
  summarise(Greedy = mean(Greedy_Coverage), Greedy_SD = sd(Greedy_Coverage),
            Random = mean(Random_Mean_Coverage), Random_SD = sd(Random_Mean_Coverage),
            Floor = mean(Theoretical_Floor), Clouds = n(), .groups = "drop")
cov_long <- bind_rows(
  cov %>% transmute(Budget_k, Method = "Greedy", Coverage = Greedy, SD = Greedy_SD),
  cov %>% transmute(Budget_k, Method = "Random", Coverage = Random, SD = Random_SD))
p2b <- ggplot(cov_long, aes(Budget_k, Coverage, colour = Method, fill = Method)) +
  geom_ribbon(aes(ymin = Coverage - SD, ymax = Coverage + SD), alpha = 0.15, colour = NA) +
  geom_line(linewidth = 0.9) +
  labs(title = "(b) Coverage by tuple budget",
       subtitle = sprintf("mean +/- 1 SD over %d point clouds", max(cov$Clouds)),
       x = "Budget k (tuples)", y = "Fraction of points covered") + theme_capri()
save_fig((p2a | p2b) + plot_annotation(caption = source_note("conditioning_roc.csv", "conditioning_greedy_vs_random.csv")),
         "conditioning", width = 10, height = 4.5)

# ---------------------------------------------------------------------------
# Syzygy residual and drift admission
# ---------------------------------------------------------------------------
exp5 <- read_result("syzygy_residual.csv") %>%
  mutate(Group = ifelse(Regime == "Mismatched", ifelse(Contains_Corrupted, "Mismatched\n(corrupted pt)",
                                                       "Mismatched\n(clean pts)"), Regime))
save_table(exp5 %>% group_by(Regime, Sub_Condition, Contains_Corrupted) %>%
             summarise(N = n(), Median_rho = median(rho), Q90_rho = quantile(rho, 0.9),
                       Median_rho_rel = median(rho_rel), .groups = "drop"), "syzygy_residual")
p3a <- ggplot(exp5, aes(Group, rho, fill = Group)) + geom_boxplot(outlier.alpha = 0.05, show.legend = FALSE) +
  scale_y_log10(labels = label_log()) +
  labs(title = expression(bold("(a) Syzygy residual ") * rho), x = NULL, y = expression(rho ~ "(log scale)")) +
  theme_capri() + theme(axis.text.x = element_text(size = 8))
exp6 <- read_result("drift_admission.csv") %>% group_by(Eta) %>%
  summarise(N = n(), Admit = sum(Admitted_Bounded), Reject = sum(Rejected_Excessive), .groups = "drop")
exp6_long <- bind_rows(
  exp6 %>% transmute(Eta, Rate = Admit / N, lo = wilson(Admit, N)$lo, hi = wilson(Admit, N)$hi,
                     What = "Bounded drift admitted (tau_exact)"),
  exp6 %>% transmute(Eta, Rate = Reject / N, lo = wilson(Reject, N)$lo, hi = wilson(Reject, N)$hi,
                     What = "Excessive drift rejected (tau_exact)"))
cal <- read_result("drift_admission_calibrated.csv")
exp6_long <- bind_rows(exp6_long,
  cal %>% transmute(Eta, Rate = 1 - False_Rejection_Bounded, lo = NA, hi = NA, What = "Bounded admitted (calibrated q95)"),
  cal %>% transmute(Eta, Rate = Rejection_Excessive, lo = NA, hi = NA, What = "Excessive rejected (calibrated q95)"))
save_table(exp6_long, "drift_admission")
p3b <- ggplot(exp6_long, aes(Eta, Rate, colour = What)) +
  geom_ribbon(data = filter(exp6_long, !is.na(lo)), aes(ymin = lo, ymax = hi, fill = What), alpha = 0.15,
              colour = NA, show.legend = FALSE) +
  geom_line(linewidth = 0.8) + scale_x_log10() + ylim(0, 1) +
  labs(title = "(b) Drift admission under the exact bound", x = expression("Deformation bound " * delta),
       y = "Rate (95% Wilson band)", colour = NULL) +
  theme_capri() + guides(colour = guide_legend(ncol = 2)) + theme(legend.text = element_text(size = 7))
save_fig((p3a | p3b) + plot_annotation(caption = source_note("syzygy_residual.csv", "drift_admission.csv",
                                                             "drift_admission_calibrated.csv")),
         "syzygy_drift", width = 11, height = 5)

# ---------------------------------------------------------------------------
# Scaling law and noise amplification
# ---------------------------------------------------------------------------
f1 <- read_result("scaling_law.csv") %>% group_by(Stratum, Codimension, Kappa) %>%
  summarise(Median_LS = median(L_S), Q25 = quantile(L_S, 0.25), Q75 = quantile(L_S, 0.75), .groups = "drop")
slopes <- f1 %>% filter(Kappa > 0) %>% group_by(Stratum) %>%
  summarise(Slope = coef(lm(log10(Median_LS) ~ log10(Kappa)))[2], .groups = "drop")
save_table(slopes, "scaling_slopes")
p4a <- ggplot(f1 %>% filter(Kappa > 0), aes(Kappa, Median_LS, colour = Stratum, fill = Stratum)) +
  geom_ribbon(aes(ymin = Q25, ymax = Q75), alpha = 0.15, colour = NA) + geom_line(linewidth = 0.8) +
  scale_x_log10(labels = label_log()) + scale_y_log10(labels = label_log()) +
  labs(title = "(a) L(S) vs distance to stratum", x = expression("Distance " * kappa),
       y = "Median L(S) (IQR band)") + theme_capri()
f10 <- read_result("noise_amplification.csv") %>% filter(J_Clean_Defined) %>%
  group_by(Stratum, Noise_Sigma) %>% summarise(Median = median(Max_J_Error), .groups = "drop")
p4b <- ggplot(f10, aes(Noise_Sigma, Median, colour = Stratum)) + geom_line(linewidth = 0.8) + geom_point(size = 1) +
  scale_x_log10(labels = label_log()) + scale_y_log10(labels = label_log()) +
  labs(title = "(b) Invariant error vs input noise", x = expression("Noise " * sigma),
       y = "Median max |J error|") + theme_capri()
save_fig((p4a | p4b) + plot_annotation(caption = source_note("scaling_law.csv", "noise_amplification.csv")),
         "scaling_noise", width = 10, height = 4.5)

# ---------------------------------------------------------------------------
# Ablation at three operating points (fixed L < 1000, calibrated L/rho, pencil kappa_J/pencil)
# ---------------------------------------------------------------------------
exp9 <- read_result("ablation.csv")
t9 <- exp9 %>% group_by(Operating_Point, Configuration) %>%
  summarise(N_Bad = sum(Tuple_Bad), N_Good = sum(!Tuple_Bad),
            TPR = mean(Accepted[!Tuple_Bad]), FPR = mean(Accepted[Tuple_Bad]), .groups = "drop") %>%
  # Display label for the fixed operating point; the CSV keeps its data label.
  mutate(Operating_Point = ifelse(Operating_Point == "manuscript", "fixed (L < 1000)", Operating_Point))
# Convention of run_ablation.py (certification view):
#   TPR = fraction of GOOD tuples accepted, FPR = fraction of BAD tuples accepted.
save_table(t9, "ablation")
p5 <- ggplot(t9 %>% pivot_longer(c(TPR, FPR), names_to = "Metric", values_to = "Rate"),
             aes(Rate, Configuration, fill = Metric)) +
  geom_col(position = "dodge") + facet_wrap(~Operating_Point) +
  labs(title = "Component ablation", x = "Acceptance rate (TPR: good tuples accepted; FPR: bad tuples accepted)", y = NULL,
       caption = source_note("ablation.csv")) + theme_capri()
save_fig(p5, "ablation", width = 9, height = 4.5)

# ---------------------------------------------------------------------------
# Trust calibration
# ---------------------------------------------------------------------------
calc <- read_result("trust_calibration_curve.csv")
save_table(calc, "trust_calibration_curve")
p6 <- ggplot(calc, aes(Predicted_Mean, Empirical_Accuracy)) +
  geom_abline(linetype = "dashed", colour = "grey60") +
  geom_errorbar(aes(ymin = Accuracy_CI_Lo, ymax = Accuracy_CI_Hi), width = 0.02) +
  geom_point(aes(shape = Reportable, size = N_Samples)) + scale_shape_manual(values = c(`TRUE` = 16, `FALSE` = 1)) +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(title = "Trust-score calibration", x = "Mean predicted confidence", y = "Empirical accuracy (95% CI)",
       shape = "Bin >= 50 samples", size = "N", caption = source_note("trust_calibration_curve.csv")) + theme_capri()
save_fig(p6, "trust_calibration", width = 6, height = 6)

# ---------------------------------------------------------------------------
# Degeneracy-detection baselines on synthetic 8-point samples
# ---------------------------------------------------------------------------
base <- read_result("true_synthetic_baselines_summary.csv") %>%
  # Display label for the fixed operating point; the CSV keeps its data label.
  mutate(Method = ifelse(Method == "capri_ls at manuscript threshold (accept if L<1000)",
                         "capri_ls at fixed threshold (accept if L<1000)", Method))
save_table(base, "synthetic_baselines")
n_samples <- max(base$N_Samples, na.rm = TRUE)
p7 <- ggplot(base %>% filter(!is.na(AUC)), aes(AUC, reorder(Method, AUC))) +
  geom_vline(xintercept = 0.5, linetype = "dashed", colour = "grey60") +
  geom_errorbarh(aes(xmin = AUC_CI_Lo, xmax = AUC_CI_Hi), height = 0.2) + geom_point(size = 2.5) +
  labs(title = "Detecting planar-degenerate 8-point samples", subtitle = sprintf("N = %d samples", n_samples),
       x = "AUC (95% CI)", y = NULL, caption = source_note("true_synthetic_baselines_summary.csv")) + theme_capri()
save_fig(p7, "synthetic_baselines", width = 7, height = 4)

# ---------------------------------------------------------------------------
# Greedy vs optimum and tuple count
# ---------------------------------------------------------------------------
f5 <- read_result("greedy_efficiency.csv")
save_table(f5 %>% group_by(N) %>% summarise(Reps = n(), Mean_Ratio = mean(Greedy_OPT_Ratio, na.rm = TRUE),
                                             Min_Ratio = if (any(is.finite(Greedy_OPT_Ratio))) min(Greedy_OPT_Ratio, na.rm = TRUE) else NA_real_,
                                             N_Below_Floor = sum(Greedy_OPT_Ratio < Theoretical_Floor, na.rm = TRUE),
                                             Floor = mean(Theoretical_Floor),
                                             MaxCov_Mean_Ratio = mean(MaxCov_OPT_Ratio, na.rm = TRUE),
                                             MaxCov_Min_Ratio = if (any(is.finite(MaxCov_OPT_Ratio))) min(MaxCov_OPT_Ratio, na.rm = TRUE) else NA_real_,
                                             MaxCov_N_Below_Floor = sum(MaxCov_OPT_Ratio < Theoretical_Floor, na.rm = TRUE),
                                             Greedy_Mean_Coverage = mean(Greedy_Coverage),
                                             MaxCov_Mean_Coverage = mean(MaxCov_Coverage), .groups = "drop"),
           "greedy_efficiency")
f5_long <- filter(f5, OPT_Method == "exhaustive") %>%
  transmute(Budget_k, Theoretical_Floor, `Penalised greedy (L(S) penalty)` = Greedy_OPT_Ratio,
            `Max coverage, kappa_J-feasible` = MaxCov_OPT_Ratio) %>%
  pivot_longer(c(-Budget_k, -Theoretical_Floor), names_to = "Selector", values_to = "Ratio")
p8a <- ggplot(f5_long, aes(factor(Budget_k), Ratio, fill = Selector)) + geom_boxplot() +
  geom_hline(aes(yintercept = mean(Theoretical_Floor)), linetype = "dashed", colour = "#d62728") +
  labs(title = "(a) Selector / exact optimum (n = 10)", x = "Budget k", y = "Coverage ratio (dashed: 1 - 1/e)") +
  theme_capri()
f8 <- read_result("tuple_count.csv") %>% group_by(Point_Count_n) %>%
  summarise(Empirical = median(Empirical_k), Theoretical = mean(Theoretical_k), .groups = "drop") %>%
  pivot_longer(-Point_Count_n, names_to = "Series", values_to = "k")
p8b <- ggplot(f8, aes(Point_Count_n, k, colour = Series)) + geom_line(linewidth = 0.8) + geom_point(size = 1) +
  labs(title = "(b) Tuples needed for full coverage", x = "Points n", y = "k (median over trials)") + theme_capri()
save_fig((p8a | p8b) + plot_annotation(caption = source_note("greedy_efficiency.csv", "tuple_count.csv")),
         "greedy_tuple_count", width = 10, height = 4.5)

# ---------------------------------------------------------------------------
# neural Confidence Fusion - reported as a negative result
# ---------------------------------------------------------------------------
fm <- read_result("fusion_metrics.csv")
save_table(fm, "fusion_metrics")
ev <- read_result("neural_fusion_eval.csv") %>% mutate(Label = ifelse(True_Label == 1, "valid (1)", "invalid (0)"))
sweep <- read_result("fusion_threshold_sweep.csv")
p9a <- ggplot(ev, aes(Probability, fill = Label)) + geom_histogram(bins = 40, position = "identity", alpha = 0.6) +
  labs(title = "(a) Predicted probability by label", x = "Neural fusion probability", y = "Count") + theme_capri()
p9b <- ggplot(sweep %>% pivot_longer(c(MCC, Balanced_Accuracy), names_to = "Metric"), aes(Threshold, value, colour = Metric)) +
  geom_line(linewidth = 0.8) + labs(title = "(b) Threshold sweep", x = "Decision threshold", y = NULL) + theme_capri()
save_fig((p9a | p9b) + plot_annotation(caption = source_note("neural_fusion_eval.csv", "fusion_threshold_sweep.csv"),
                                       subtitle = "Probabilities span a narrow band; no threshold beats the majority baseline by a useful margin."),
         "neural_fusion", width = 10, height = 4.5)

# ---------------------------------------------------------------------------
# Shape prior training (ml_models/shape_prior/train.py)
# ---------------------------------------------------------------------------
sp <- read_result("shape_prior.csv") %>% pivot_longer(c(Train_Loss, Val_Loss), names_to = "Split", values_to = "Loss")
sph <- read_result("shape_prior_eval_hist.csv")
# R^2 over the coordinates that vary (constant homogeneous coordinates excluded),
# and the squared Pearson correlation over all coordinates (not an R^2).
sps <- read_result("shape_prior_eval_scatter.csv")
varying <- sps %>% group_by(Coord_Idx) %>% summarise(v = var(True_Val), .groups = "drop") %>% filter(v > 1e-12)
spv <- sps %>% filter(Coord_Idx %in% varying$Coord_Idx)
save_table(tibble(
  N_Samples = n_distinct(sps$Sample_Idx), N_Coordinates = n_distinct(sps$Coord_Idx),
  N_Varying_Coordinates = nrow(varying),
  R2_Varying = 1 - sum((spv$True_Val - spv$Pred_Val)^2) / sum((spv$True_Val - mean(spv$True_Val))^2),
  Squared_Correlation_All = cor(sps$True_Val, sps$Pred_Val)^2,
  Median_Sample_MSE = median(sph$MSE_Error)), "shape_prior")
p10a <- ggplot(sp, aes(Epoch, Loss, colour = Split)) + geom_line(linewidth = 0.8) +
  labs(title = "(a) Shape prior learning curve", y = "MSE") + theme_capri()
p10b <- ggplot(sph, aes(MSE_Error)) + geom_histogram(bins = 50, fill = "#1f77b4") + scale_x_log10() +
  labs(title = "(b) Validation per-sample error", x = "MSE (log)", y = "Count") + theme_capri()
save_fig((p10a | p10b) + plot_annotation(caption = source_note("shape_prior.csv", "shape_prior_eval_hist.csv")),
         "shape_prior", width = 10, height = 4.5)

# ---------------------------------------------------------------------------
# Runtime appendix (exp12)
# ---------------------------------------------------------------------------
rt <- read_result("runtime.csv")
save_table(rt %>% select(Points, Stage, Language, Median_ms) %>% pivot_wider(names_from = Language, values_from = Median_ms) %>%
             mutate(Speedup = Python / `C++`), "runtime")
p11 <- ggplot(rt, aes(Points, Median_ms, colour = Language)) + geom_line() + geom_point(size = 1) +
  facet_wrap(~Stage, scales = "free_y") + scale_y_log10() +
  labs(title = "Runtime by stage", y = "Median time (ms, log)", caption = source_note("runtime.csv")) + theme_capri()
save_fig(p11, "runtime", width = 9, height = 5)

# ---------------------------------------------------------------------------
# Runtime of the pencil and kappa_J computations
# ---------------------------------------------------------------------------
rt2 <- read_result("pencil_runtime.csv")
save_table(rt2 %>% select(Stage, N_Points, Work_Items, Implementation, Median_ms, IQR_ms, Outputs_Agree), "pencil_runtime")
p15 <- ggplot(rt2, aes(Median_ms, interaction(Implementation, N_Points, sep = ", n = "), fill = Implementation)) +
  geom_col(show.legend = FALSE) + facet_wrap(~Stage, scales = "free", ncol = 1) + scale_x_log10() +
  labs(title = "Runtime of kappa_J, the pencil statistic and the selector", x = "Median time (ms, log)", y = NULL,
       caption = source_note("pencil_runtime.csv")) + theme_capri()
save_fig(p15, "pencil_runtime", width = 8, height = 7)

# ---------------------------------------------------------------------------
# Pencil drift test vs alternatives (frame-free)
# ---------------------------------------------------------------------------
p6b <- read_result("pencil_drift_summary.csv")
save_table(p6b, "pencil_drift")
d6b <- read_result("pencil_drift.csv")
roc_points <- function(y, s) {
  o <- order(-s); y <- y[o]
  data.frame(FPR = c(0, cumsum(1 - y) / sum(1 - y)), TPR = c(0, cumsum(y) / sum(y)))
}
roc6b <- bind_rows(lapply(c("pencil_q10", "registration_rms", "median_tuple_dJ", "single_tuple_dJ"), function(m)
  roc_points(as.numeric(d6b$Label_Excessive), d6b[[m]]) %>% mutate(Method = m)))
p13a <- ggplot(roc6b, aes(FPR, TPR, colour = Method)) + geom_line(linewidth = 0.8) +
  geom_abline(linetype = "dashed", colour = "grey60") + coord_equal() +
  labs(title = "(a) Bounded vs excessive drift", x = "False positive rate", y = "True positive rate") + theme_capri()
p13b <- ggplot(p6b, aes(TPR_at_5pct_FPR, reorder(Method, TPR_at_5pct_FPR))) +
  geom_errorbarh(aes(xmin = TPR_CI_Lo, xmax = TPR_CI_Hi), height = 0.2) + geom_point(size = 2.5) +
  labs(title = "(b) TPR at 5% FPR (95% bootstrap CI)", x = "TPR", y = NULL) + theme_capri()
save_fig((p13a | p13b) + plot_annotation(caption = source_note("pencil_drift.csv", "pencil_drift_summary.csv"),
                                         subtitle = sprintf("Frame-free drift, N = %d scenes", nrow(d6b))),
         "pencil_drift", width = 11, height = 4.8)

# ---------------------------------------------------------------------------
# Conditioning scores vs real noise error
# ---------------------------------------------------------------------------
k4b <- read_result("kappa_conditioning.csv")
s4b <- read_result("kappa_conditioning_summary.csv")
save_table(s4b, "kappa_conditioning")
k4b_long <- bind_rows(
  k4b %>% transmute(Score = "kappa_J", x = log10(Kappa_J), y = Target_Median_Max_dlogJ),
  k4b %>% transmute(Score = "L(S)", x = Log_L_S / log(10), y = Target_Median_Max_dlogJ))
labs4b <- s4b %>% filter(Score %in% c("Kappa_J", "Log_L_S")) %>%
  mutate(Score = ifelse(Score == "Kappa_J", "kappa_J", "L(S)"), lab = sprintf("Spearman %.3f", Spearman))
p14 <- ggplot(k4b_long, aes(x, y)) + geom_point(alpha = 0.25, size = 0.8) + scale_y_log10(labels = label_log()) +
  facet_wrap(~Score, scales = "free_x") +
  geom_text(data = labs4b, aes(x = -Inf, y = Inf, label = lab), hjust = -0.1, vjust = 1.5, inherit.aes = FALSE) +
  labs(title = "Conditioning score vs noise error of log J (sigma = 1e-3)", x = "log10 score",
       y = "median max |d log J|", caption = source_note("kappa_conditioning.csv")) + theme_capri()
save_fig(p14, "kappa_vs_L", width = 9, height = 4.5)

# ---------------------------------------------------------------------------
# Forensic validation fixtures (S1, S6a, S6b): code checks, not detection results
# ---------------------------------------------------------------------------
FOR <- "forensics"
if (file.exists(file.path(RES, FOR, "rendered_scenes.csv"))) {
  s1 <- read_result(file.path(FOR, "synthetic_segments.csv"))
  save_table(s1, "forensics_null")
  s1r <- read_result(file.path(FOR, "rendered_scenes.csv"))
  pf1 <- ggplot(s1r, aes(factor(Distortion_k1), Mean_z2)) + geom_boxplot() + geom_hline(yintercept = 1, linetype = "dashed") +
    scale_y_log10() + labs(title = "(a) S1: rendered scenes", x = "Radial distortion k1", y = "Mean z^2 per image") + theme_capri()
  s6a <- read_result(file.path(FOR, "multiview_synthetic.csv"))
  pf2 <- ggplot(s6a %>% filter(Check != "i_noise_free"),
                aes(ifelse(Check == "iii_injection", Offset, 0), Pencil_Q10, colour = factor(Noise_px))) +
    geom_point(alpha = 0.6) + stat_summary(fun = median, geom = "line") +
    labs(title = "(b) S6a: injected displacement", x = "Displacement (scene units)", y = "Pencil Q0.1 |d log|CR||",
         colour = "Noise px") + theme_capri()
  s6b <- read_result(file.path(FOR, "video_fixture.csv")) %>% filter(Used)
  pf3 <- ggplot(s6b, aes(factor(Warp_Amplitude), Statistic)) + geom_boxplot() + geom_jitter(width = 0.1, alpha = 0.5) +
    labs(title = "(c) S6b: rendered clips", x = "Time-varying warp amplitude", y = "Window statistic (pencil Q0.1)") + theme_capri()
  save_fig((pf1 | pf2 | pf3) + plot_annotation(
    caption = source_note("forensics/rendered_scenes.csv", "forensics/multiview_synthetic.csv", "forensics/video_fixture.csv"),
    subtitle = "Implementation checks on simulated fixtures."),
    "forensic_fixtures", width = 13, height = 4.5)
}

# ---------------------------------------------------------------------------
# Pre-registered components: coplanarity score, logistic fusion, calibrated trust
# (simulation study; scopes in docs/prereg/*.md)
# ---------------------------------------------------------------------------
e14 <- read_result("degeneracy_matched_summary.csv") %>% filter(Rule == "primary", Label == "k>=6")
save_table(e14 %>% select(Arm, Method, AUC, AUC_CI95_Lo, AUC_CI95_Hi, TPR_Test_at_Cal_Threshold, FPR_Test_at_Cal_Threshold, N_Test),
           "degeneracy_matched")
e8 <- read_result("logistic_fusion_metrics.csv") %>% filter(subset == "all", metric == "mcc", !grepl("@", rule))
save_table(e8, "logistic_fusion")
e11 <- read_result("trust_calibrated_metrics.csv") %>% filter(split %in% c("test", "shifted"), label == "eps_median<=0.1")
save_table(e11, "trust_calibrated")
pa <- ggplot(e14, aes(AUC, reorder(paste(Arm, Method), AUC))) +
  geom_errorbarh(aes(xmin = AUC_CI95_Lo, xmax = AUC_CI95_Hi), height = 0.2) + geom_point(size = 2) +
  labs(title = "(a) A: degeneracy, k >= 6", subtitle = "Coplanarity score: pre-registered representation only",
       x = "AUC (95% CI)", y = NULL) + theme_capri()
pb <- ggplot(e8 %>% filter((arm == "primary" & rule %in% c("C0", "C1*", "C2", "C3", "M1", "M1-lin", "M1-no-rho")) |
                             (arm == "D" & rule %in% c("C0-D", "C1*-D", "C3-D", "M1-D"))),
             aes(estimate, reorder(paste(arm, rule), estimate))) +
  geom_errorbarh(aes(xmin = ci_low, xmax = ci_high), height = 0.2) + geom_point(size = 2) +
  labs(title = "(b) B: logistic fusion, MCC", subtitle = "Primary arm needs a common known chart", x = "MCC (95% CI)", y = NULL) + theme_capri()
rel <- read_result("trust_calibrated_reliability.csv") %>% filter(binning == "equal_width", n_items > 0, score %in% c("R_raw", "R_cal"))
pc <- ggplot(rel, aes(mean_prediction, observed_frequency, colour = score)) +
  geom_abline(linetype = "dashed", colour = "grey60") + geom_errorbar(aes(ymin = ci_low, ymax = ci_high), width = 0.02) +
  geom_point(aes(size = n_items)) + coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(title = "(c) C: trust reliability (test)", subtitle = "Marginal only; fails within strata and under shift",
       x = "Mean prediction", y = "Observed frequency", size = "N") + theme_capri()
save_fig((pa | pb | pc) + plot_annotation(
  caption = source_note("degeneracy_matched_summary.csv", "logistic_fusion_metrics.csv", "trust_calibrated_reliability.csv"),
  subtitle = "Simulation study."),
  "components", width = 16, height = 6)

# ---------------------------------------------------------------------------
# Synthetic evaluation (docs/prereg/evaluation_plan.md): detector benchmark, dose-response, representation robustness
# Families are shown with descriptive labels (synthetic_data/injection.py).
# ---------------------------------------------------------------------------
if (file.exists(file.path(RES, "detector_benchmark_cells.csv"))) {
  cells <- read_result("detector_benchmark_cells.csv")
  keep <- c("pencil_q10", "acat", "gauge_q90", "deltaj_med", "registration_rms", "split_glrt", "sampson_mean")
  stat_names <- c(pencil_q10 = "Pencil Q0.1", acat = "Standardized pencils + Cauchy",
                  gauge_q90 = "Additive gauge fit", deltaj_med = "Tuple Delta J (median)",
                  registration_rms = "Projective registration", split_glrt = "Split GLRT (known sigma)",
                  sampson_mean = "Sampson error")
  family_names <- c(global_projective = "Global projective map", jitter = "Point jitter", drift = "Coherent drift",
                    warp = "Local warp", swap = "Track swaps")
  family_cols <- c("Global projective map" = "grey55", "Point jitter" = "#0072B2",
                   "Coherent drift" = "#E69F00", "Local warp" = "#009E73", "Track swaps" = "#CC79A7")
  family_scale <- scale_colour_manual(values = family_cols, limits = names(family_cols), drop = FALSE)
  dstar <- tibble(Family = c("jitter", "drift", "warp", "swap"), Dose = c(2, 4, 2, 0.05))
  e6 <- cells %>% filter(Sigma == 0.5, Statistic %in% keep) %>% inner_join(dstar, by = c("Family", "Dose"))
  save_table(e6 %>% select(Family, Dose, Statistic, AUC_vs_rigid, Reject_ITT, Coverage), "detector_benchmark")
  e6 <- e6 %>% mutate(Injection = factor(family_names[Family], levels = family_names),
                      Stat = factor(stat_names[Statistic], levels = rev(stat_names[keep])))
  q1 <- ggplot(e6, aes(AUC_vs_rigid, Stat, colour = Injection)) +
    geom_point(size = 2.5, position = position_dodge(width = 0.5)) +
    labs(title = "(a) Detector benchmark: AUC at the detectable dose",
         subtitle = "sigma = 0.5 px; dose 2 / 4 / 2 px, 5% swaps",
         x = "AUC vs consistent windows", y = NULL, colour = "Injection") + family_scale + theme_capri()
  dose <- cells %>% filter(Sigma == 0.5, Statistic == "pencil_q10", Family %in% c("global_projective", "jitter", "drift", "warp")) %>%
    mutate(Injection = factor(family_names[Family], levels = family_names))
  q2 <- ggplot(dose, aes(Dose, Reject_ITT, colour = Injection)) + geom_line() + geom_point() +
    geom_hline(yintercept = 0.05, linetype = "dashed", colour = "grey60") + scale_x_log10() +
    labs(title = "(b) Dose-response of the pencil statistic",
         subtitle = "Global projective map = provable blind spot",
         x = "Induced displacement (px, log)", y = "Rejection rate (5% threshold)", colour = "Injection") + family_scale + guides(colour = "none") + theme_capri()
  rep_names <- c(H1_R0 = "Unit columns, largest-triangle base (registered)", H1_R1 = "Affine representatives X/X4",
                 H1_R2 = "Fixed random projective frame", H1_R3 = "First-three base triple",
                 H1_R4 = "Smallest-triangle base triple", H1_R5 = "Random base triple")
  e7 <- read_result("coplanarity_robustness_comparisons.csv") %>% filter(grepl("^H1_", Hypothesis)) %>%
    mutate(Representation = rep_names[Hypothesis])
  q3 <- ggplot(e7, aes(Difference, reorder(Representation, Difference))) +
    geom_vline(xintercept = 0, linetype = "dashed", colour = "grey60") +
    geom_errorbarh(aes(xmin = CI_Lo, xmax = CI_Hi), height = 0.2) + geom_point(size = 2.5) +
    labs(title = "(c) Representation robustness", subtitle = "Coplanarity score minus DEGENSAC inlier count",
         x = "AUC difference (95% / 99% CI)", y = NULL) + theme_capri()
  save_fig((q1 | q2 | q3) + plot_layout(guides = "collect") + plot_annotation(
    caption = source_note("detector_benchmark_cells.csv", "coplanarity_robustness_comparisons.csv"),
    subtitle = "Simulation study. Pre-registered evaluation plan.") &
    theme(legend.position = "bottom"),
    "evaluation", width = 18, height = 5.5)
}

# ---------------------------------------------------------------------------
# Stratification confusion matrices: threshold vs rank classifier per noise level
# ---------------------------------------------------------------------------
strata_levels <- c("generic", "boundary", "compound", "critical")
conf <- read_result("stratification_classifiers.csv") %>%
  pivot_longer(c(Threshold_Classifier, Rank_Classifier), names_to = "Classifier", values_to = "Predicted") %>%
  mutate(Classifier = recode(Classifier, Threshold_Classifier = "Single threshold", Rank_Classifier = "Rank classifier"),
         Noise = factor(ifelse(Noise_Sigma == 0, "noise-free", paste0("sigma = ", format(Noise_Sigma, scientific = TRUE))),
                        levels = c("noise-free", "sigma = 1e-06", "sigma = 1e-04"))) %>%
  count(Classifier, Noise, True_Stratum, Predicted) %>%
  group_by(Classifier, Noise, True_Stratum) %>% mutate(Share = n / sum(n)) %>% ungroup() %>%
  mutate(True_Stratum = factor(True_Stratum, levels = rev(strata_levels)), Predicted = factor(Predicted, levels = strata_levels))
p_conf <- ggplot(conf, aes(Predicted, True_Stratum, fill = Share)) + geom_tile(colour = "white") +
  geom_text(aes(label = percent(Share, accuracy = 1)), size = 2.8) +
  facet_grid(Classifier ~ Noise) + scale_fill_gradient(low = "#f4f1ea", high = "#e56b2f", labels = percent, breaks = c(0, 0.5, 1)) +
  labs(title = "Stratum classification: confusion matrices", x = "Predicted stratum", y = "True stratum",
       fill = "Row share", caption = source_note("stratification_classifiers.csv")) + theme_capri() +
  theme(panel.grid = element_blank(), axis.text.x = element_text(angle = 30, hjust = 1))
save_fig(p_conf, "stratification_confusion", width = 8, height = 5)

# ---------------------------------------------------------------------------
# Neural fusion: confusion matrix at 0.5 and ROC curve
# ---------------------------------------------------------------------------
fe <- read_result("neural_fusion_eval.csv") %>% mutate(Predicted = as.integer(Probability >= 0.5))
fcm <- fe %>% count(True_Label, Predicted) %>% complete(True_Label = 0:1, Predicted = 0:1, fill = list(n = 0)) %>%
  mutate(True_Label = factor(True_Label, levels = 1:0, labels = c("valid (1)", "invalid (0)")),
         Predicted = factor(Predicted, levels = 0:1, labels = c("invalid (0)", "valid (1)")))
pf1 <- ggplot(fcm, aes(Predicted, True_Label, fill = n)) + geom_tile(colour = "white") + geom_text(aes(label = n)) +
  scale_fill_gradient(low = "#f4f1ea", high = "#142536") + guides(fill = "none") +
  labs(title = "(a) Confusion matrix at p >= 0.5", x = "Predicted", y = "True") + theme_capri() + theme(panel.grid = element_blank())
roc_points <- function(score, label) {
  th <- sort(unique(c(Inf, score)), decreasing = TRUE)
  tibble(FPR = sapply(th, function(t) mean(score[label == 0] >= t)), TPR = sapply(th, function(t) mean(score[label == 1] >= t)))
}
froc <- roc_points(fe$Probability, fe$True_Label)
pf2 <- ggplot(froc, aes(FPR, TPR)) + geom_abline(linetype = "dashed", colour = "grey60") + geom_step(direction = "vh") +
  coord_equal() + labs(title = "(b) ROC curve", x = "False-positive rate", y = "True-positive rate") + theme_capri()
save_fig((pf1 | pf2) + plot_annotation(caption = source_note("neural_fusion_eval.csv")), "neural_fusion_confusion", width = 8, height = 3.8)

# ---------------------------------------------------------------------------
# Null calibration: size at nominal 5% by noise level
# ---------------------------------------------------------------------------
sizes <- read_result("null_calibration_sizes.csv") %>% filter(!is.na(Sigma), !grepl("misspecified|half", Test))
p_null <- ggplot(sizes, aes(factor(Sigma), Size_0.05, colour = Test, group = Test)) +
  annotate("rect", xmin = -Inf, xmax = Inf, ymin = 0.035, ymax = 0.065, alpha = 0.15, fill = "#2f6f4f") +
  geom_hline(yintercept = 0.05, linetype = "dashed", colour = "grey50") +
  geom_pointrange(aes(ymin = CI_lo_0.05, ymax = CI_hi_0.05), position = position_dodge(width = 0.5)) +
  scale_y_continuous(labels = percent) +
  labs(title = "Null calibration: false-alarm rate at nominal 5%", subtitle = "Shaded: pre-registered band 3.5-6.5%",
       x = "Pixel noise sigma (px)", y = "Rejection rate on consistent windows", colour = NULL,
       caption = source_note("null_calibration_sizes.csv")) + theme_capri() + guides(colour = guide_legend(ncol = 2))
save_fig(p_null, "null_calibration", width = 8, height = 4.8)

# ---------------------------------------------------------------------------
# Pre-registered replication (docs/prereg/consistency_test.md): combination power and sizes
# ---------------------------------------------------------------------------
cv2_power <- read_result("consistency_test_power.csv")
cv2_dec <- read_result("consistency_test_decisions.csv")
cv2_size <- as.numeric(cv2_dec$Value[grepl("^H3 size", cv2_dec$Item)])
cv2_long <- bind_rows(
  transmute(cv2_power, Type, Statistic = "Pencil Q0.1", Power = Power_pencil),
  transmute(cv2_power, Type, Statistic = "Sampson error", Power = Power_sampson),
  transmute(cv2_power, Type, Statistic = "Cauchy combination", Power = Power_combined)) %>%
  mutate(lo = wilson(Power * 400, 400)$lo, hi = wilson(Power * 400, 400)$hi,
         Type = factor(Type, levels = c("jitter", "drift", "warp", "swap")),
         Statistic = factor(Statistic, levels = c("Pencil Q0.1", "Sampson error", "Cauchy combination")))
p_cv2 <- ggplot(cv2_long, aes(Type, Power, fill = Statistic)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.75) +
  geom_errorbar(aes(ymin = lo, ymax = hi), position = position_dodge(width = 0.8), width = 0.25, colour = "grey30") +
  scale_y_continuous(labels = percent, limits = c(0, 1)) +
  scale_fill_manual(values = c("#7a9cc6", "#c9a66b", "#2f6f4f")) +
  labs(title = "Combining the pencil statistic with the Sampson error",
       subtitle = paste0("Nominal 5%; 400 windows per type; 95% Wilson intervals. Combination size: ",
                         percent(cv2_size, accuracy = 0.01)),
       x = "Injection type (detectable dose)", y = "Power", fill = NULL,
       caption = source_note("consistency_test_power.csv", "consistency_test_decisions.csv")) + theme_capri()
save_fig(p_cv2, "consistency_test_combination", width = 7, height = 4.2)
save_table(cv2_power, "consistency_test_power")
save_table(read_result("consistency_test_sizes.csv"), "consistency_test_sizes")
save_table(read_result("consistency_test_decisions.csv"), "consistency_test_decisions")
save_table(read_result("consistency_test_misspecification.csv"), "consistency_test_misspecification")

# ---------------------------------------------------------------------------
# Tables: clip aggregation and selection by information
# ---------------------------------------------------------------------------
save_table(read_result("clip_aggregation.csv"), "clip_aggregation")
save_table(read_result("selection_by_information_summary.csv"), "selection_by_information")

# ---------------------------------------------------------------------------
# Pipeline architecture. A diagram, not
# data. The stages follow pipeline/main_pipeline.py: run_pipeline; the CAPRI layer
# run_capri_reliability_layer (steps 1-3, in its order: kappa_J-feasible tuples,
# max-coverage selection, rank stratum of each selected tuple); and
# cross_pair_pencil_consistency (step 4; the window-level planar abstention is
# the one in experiments/eval/mv_window.py).
# ---------------------------------------------------------------------------
# The labels use Greek letters, which R drops in a non-UTF-8 locale.
if (!isTRUE(l10n_info()[["UTF-8"]])) invisible(suppressWarnings(Sys.setlocale("LC_CTYPE", "C.UTF-8")))
diagram_font <- "DejaVu Sans"   # has the Greek glyphs used below
navy <- "#142536"; orange <- "#e56b2f"; muted <- "#4f5a64"; card <- "#f4f1ea"
diagram_box <- function(x, y, w, h, title, detail, fill = card, colour = "grey55",
                        linetype = "solid", title_colour = navy) {
  list(rect = data.frame(xmin = x, xmax = x + w, ymin = y, ymax = y + h, fill = fill,
                         colour = colour, linetype = linetype),
       text = data.frame(x = x + w / 2, y = c(y + h - 2.6, y + (h - 4.4) / 2), label = c(title, detail),
                         face = c("bold", "plain"), size = c(3.0, 2.55), colour = c(title_colour, muted)))
}
sfm <- list(c("Images", "two view\nsubsets V_A, V_B"), c("Matching", "SIFT features,\nimage pairs"),
            c("F matrix", "RANSAC"), c("Cameras", "P1 = [I | 0],\nP2 from F"),
            c("Triangulation", "DLT, one frame\nper image pair"), c("Tyler frame", "canonical,\nscale-free"))
capri <- list(c("1  Conditioning", "feasible tuples:\nκ_J below the\ncalibrated threshold"),
              c("2  Selection", "greedy maximum\ncoverage,\n≥ (1 − 1/e) OPT"),
              c("3  Stratum", "rank classifier on each\nselected tuple;\nreject critical tuples"),
              c("4  Pencil test", "two reconstructions;\nabstain if (near) planar;\nS = Q_0.1 of |Δ log|CR||"))
parts <- list(); arrows <- data.frame()
w1 <- 14.8; g1 <- (100 - 6 * w1) / 5
for (i in seq_along(sfm)) {
  x <- (i - 1) * (w1 + g1)
  parts[[length(parts) + 1]] <- diagram_box(x, 47, w1, 12, sfm[[i]][1], sfm[[i]][2])
  if (i > 1) arrows <- rbind(arrows, data.frame(x = x - g1 + 0.25, xend = x - 0.25, y = 53, yend = 53))
}
w2 <- 22.5; g2 <- (97 - 4 * w2) / 3
for (i in seq_along(capri)) {
  x <- 1.5 + (i - 1) * (w2 + g2)
  parts[[length(parts) + 1]] <- diagram_box(x, 19, w2, 16, capri[[i]][1], capri[[i]][2],
                                            fill = "white", colour = orange, title_colour = orange)
  if (i > 1 && i < 4) arrows <- rbind(arrows, data.frame(x = x - g2 + 0.25, xend = x - 0.25, y = 27, yend = 27))
}
parts[[length(parts) + 1]] <- diagram_box(1.5, 0, 26, 10.5, "Certified points", "a match is kept if its point\nis in an accepted tuple")
parts[[length(parts) + 1]] <- diagram_box(35, 0, 26, 10.5, "Dense MVS", "downstream and unchanged;\nnot part of this repository",
                                          fill = "white", linetype = "dashed")
parts[[length(parts) + 1]] <- diagram_box(72.5, 0, 26, 10.5, "Diagnostics", "per-point pencil scores,\nabstentions, statistic S")
arrows <- rbind(arrows,
  data.frame(x = 92.6, xend = 92.6, y = 46.7, yend = 42.3),    # Tyler frame -> CAPRI layer
  data.frame(x = 14.5, xend = 14.5, y = 13.9, yend = 10.8),    # accepted tuples -> certified points
  data.frame(x = 85.5, xend = 85.5, y = 13.9, yend = 10.8),    # pencil test -> diagnostics
  data.frame(x = 27.8, xend = 34.7, y = 5.25, yend = 5.25))    # certified points -> dense MVS
rects <- do.call(rbind, lapply(parts, `[[`, "rect")); texts <- do.call(rbind, lapply(parts, `[[`, "text"))
p18 <- ggplot() +
  annotate("text", x = 0, y = 61.5, hjust = 0, size = 3.2, fontface = "bold", colour = navy, family = diagram_font,
           label = "Sparse reconstruction (standard stages)") +
  annotate("rect", xmin = 0, xmax = 100, ymin = 14.2, ymax = 42, fill = "#fdf3ec", colour = orange, linewidth = 0.8) +
  annotate("text", x = 1.5, y = 39.3, hjust = 0, size = 3.2, fontface = "bold", colour = orange, family = diagram_font,
           label = "CAPRI layer") +
  annotate("text", x = 98.5, y = 39.3, hjust = 1, size = 2.45, colour = muted, family = diagram_font,
           label = "steps 1–3 inside one reconstruction; step 4 compares two") +
  annotate("text", x = 50, y = 16.3, size = 2.45, colour = muted, fontface = "italic", family = diagram_font,
           label = "No ρ gate inside one reconstruction: ρ(S) = 0 there.") +
  geom_rect(data = rects, aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = I(fill),
                              colour = I(colour), linetype = I(linetype)), linewidth = 0.45) +
  geom_text(data = texts, aes(x, y, label = label, fontface = face, size = I(size), colour = I(colour)),
            lineheight = 0.92, family = diagram_font) +
  geom_segment(data = arrows, aes(x = x, xend = xend, y = y, yend = yend), colour = navy, linewidth = 0.5,
               arrow = arrow(length = unit(0.11, "cm"), type = "closed")) +
  coord_cartesian(xlim = c(0, 100), ylim = c(-1, 63.5), expand = FALSE, clip = "off") +
  labs(caption = "Source: pipeline/main_pipeline.py. End-to-end runs on synthetic scenes only.") +
  theme_void(base_size = 10, base_family = diagram_font) +
  theme(plot.caption = element_text(colour = "grey40", size = 7.5), plot.margin = margin(4, 4, 4, 4))
ggsave(file.path(OUT, "pipeline.pdf"), p18, width = 7.2, height = 4.6, device = cairo_pdf)
ggsave(file.path(OUT, "pipeline.png"), p18, width = 7.2, height = 4.6, dpi = 250, bg = "white")
cat("  wrote figures/pipeline.{pdf,png}\n")

cat("Done.\n")
