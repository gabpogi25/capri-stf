# figures/paper/render_assets.R
# Renders the image assets of the illustrative paper figures (figures/paper/*.template.svg are the
# Inkscape layouts that place them) and fills the numbers of the layouts from the result files.
# Data: experiments/results/figure_*.csv (experiments/run_figure_examples.py) and
# abstract_photo_*.csv (experiments/run_abstract_photo_demo.py). Photo overlays shared with the
# graphical abstracts are linked from figures/abstracts/assets/.
# Run from the repository root: Rscript figures/paper/render_assets.R

suppressPackageStartupMessages({ library(ggplot2); library(dplyr) })

RES <- "experiments/results"
OUT <- "figures/paper/assets"
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
read_result <- function(f) read.csv(file.path(RES, f), comment.char = "#")
COL_A <- "#1f77ff"; COL_B <- "#ff2a1a"; GREY <- "#9aa0a6"; INK <- "#202124"
FAM_COL <- c(jitter = "#e8710a", drift = "#9334e6", warp = "#188038", swap = "#d01884", global_projective = "#1a73e8")
EXAG <- 4                                            # display exaggeration of the injected displacements

save_png <- function(g, file, w, h) ggsave(file.path(OUT, file), g, width = w, height = h, dpi = 300, bg = "transparent")
blank <- theme_void() + theme(plot.background = element_rect(fill = "transparent", colour = NA))
rot <- function(az, el) {
  a <- az * pi / 180; e <- el * pi / 180
  Ry <- matrix(c(cos(a), 0, sin(a), 0, 1, 0, -sin(a), 0, cos(a)), 3, byrow = TRUE)
  Rx <- matrix(c(1, 0, 0, 0, cos(e), -sin(e), 0, sin(e), cos(e)), 3, byrow = TRUE)
  Rx %*% Ry
}
view <- function(xyz, V, centre = c(0, 0, 0)) {
  p <- t(V %*% (t(as.matrix(xyz)) - centre))
  data.frame(u = p[, 1], v = p[, 2], depth = p[, 3])
}

# ------------------------------------------------------------------ teaser: two frames
frames <- read_result("figure_teaser_frames.csv")
model <- read_result("abstract_photo_model.csv") %>% select(Point, R, G, B)
frames <- inner_join(frames, model, by = "Point")
cloud <- function(sub, file, az, el, flip = 1) {
  d <- frames %>% filter(Subset == sub)
  keep <- Reduce(`&`, lapply(c("X", "Y", "Z"), function(k) d[[k]] > quantile(d[[k]], 0.02) & d[[k]] < quantile(d[[k]], 0.98)))
  d <- d[keep, ]
  centre <- sapply(d[, c("X", "Y", "Z")], median)
  P <- cbind(d, view(d[, c("X", "Y", "Z")], rot(az, el), centre))
  P <- P[order(-P$depth), ]
  g <- ggplot(P, aes(u, flip * v)) + geom_point(colour = rgb(P$R, P$G, P$B, maxColorValue = 255) |> adjustcolor(red.f = 0.75, green.f = 0.75, blue.f = 0.75), size = 1.9) +
    coord_equal() + blank
  save_png(g, file, 4, 3.2)
}
cloud("A", "teaser_cloud_A.png", 0, 20, -1)           # COLMAP frame: y points down
cloud("B", "teaser_cloud_B.png", 35, -25, 1)

# ------------------------------------------------------------------ design: scene and cameras
scene <- read_result("figure_design_scene.csv")
V <- rot(20, 28)
pts <- cbind(scene %>% filter(Kind == "point"), view(scene %>% filter(Kind == "point") %>% select(X, Y, Z), V))
cams <- scene %>% filter(Kind == "camera")
pyr <- list()
for (k in seq_len(nrow(cams))) {                     # small pyramid pointing at the origin
  C <- unlist(cams[k, c("X", "Y", "Z")]); f <- -C / sqrt(sum(C^2))
  up <- c(0, 1, 0); r <- cross <- c(f[2] * up[3] - f[3] * up[2], f[3] * up[1] - f[1] * up[3], f[1] * up[2] - f[2] * up[1])
  r <- r / sqrt(sum(r^2)); u <- c(r[2] * f[3] - r[3] * f[2], r[3] * f[1] - r[1] * f[3], r[1] * f[2] - r[2] * f[1])
  corners <- t(sapply(list(c(1, 1), c(-1, 1), c(-1, -1), c(1, -1)), function(s) C + 0.9 * f + 0.45 * s[1] * r + 0.28 * s[2] * u))
  Q <- view(rbind(C, corners), V)
  idx <- rbind(c(1, 2), c(1, 3), c(1, 4), c(1, 5), c(2, 3), c(3, 4), c(4, 5), c(5, 2))
  pyr[[k]] <- data.frame(x = Q$u[idx[, 1]], y = Q$v[idx[, 1]], xend = Q$u[idx[, 2]], yend = Q$v[idx[, 2]],
                         Subset = cams$Subset[k])
}
pyr <- bind_rows(pyr)
cube <- expand.grid(x = c(-1, 1), y = c(-1, 1), z = c(-1, 1))
edges <- do.call(rbind, lapply(seq_len(8), function(i) do.call(rbind, lapply(seq_len(8), function(j)
  if (i < j && sum(cube[i, ] != cube[j, ]) == 1) data.frame(i = i, j = j)))))
cv <- view(cube, V)
g <- ggplot() +
  geom_segment(data = data.frame(x = cv$u[edges$i], y = cv$v[edges$i], xend = cv$u[edges$j], yend = cv$v[edges$j]),
               aes(x, y, xend = xend, yend = yend), colour = "#dadce0", linewidth = 0.4) +
  geom_point(data = pts, aes(u, v), colour = INK, size = 1.2) +
  geom_segment(data = pyr, aes(x, y, xend = xend, yend = yend, colour = Subset), linewidth = 0.8, lineend = "round") +
  scale_colour_manual(values = c(A = COL_A, B = COL_B), guide = "none") + coord_equal() + blank
save_png(g, "design_scene.png", 4.2, 3.4)

# ------------------------------------------------------------------ design: one view per injection family
win <- read_result("figure_design_window.csv") %>% mutate(Moved = as.logical(Moved))
ends <- win %>% mutate(k = ifelse(Family == "swap", 1, EXAG), xe = X0 + k * (X1 - X0), ye = Y0 + k * (Y1 - Y0))
box <- c(range(c(ends$X0, ends$xe)), range(c(ends$Y0, ends$ye)))
pad <- 0.04 * max(diff(box[1:2]), diff(box[3:4]))
xl <- c(box[1] - pad, box[2] + pad); yl <- c(box[3] - pad, box[4] + pad)
for (fam in names(FAM_COL)) {
  d <- win %>% filter(Family == fam)
  k <- if (fam == "swap") 1 else EXAG
  m <- d %>% filter(Moved) %>% mutate(xe = X0 + k * (X1 - X0), ye = Y0 + k * (Y1 - Y0))
  g <- ggplot() +
    geom_point(data = d, aes(X0, -Y0), colour = GREY, size = 1.3) +
    geom_segment(data = m, aes(X0, -Y0, xend = xe, yend = -ye), colour = FAM_COL[[fam]], linewidth = 0.55,
                 arrow = arrow(length = unit(0.035, "inches"), type = "closed", angle = 25)) +
    geom_point(data = m, aes(X0, -Y0), colour = FAM_COL[[fam]], size = 1.4) +
    coord_equal(xlim = xl, ylim = -rev(yl), expand = FALSE) + blank
  save_png(g, paste0("design_", fam, ".png"), 2.4, 2.4 * diff(yl) / diff(xl))
}

# ------------------------------------------------------------------ pencil of planes (schematic, exact geometry)
# The line p_i p_j is the y-axis; four half-planes through it, each containing one point.
theta <- c(-50, -15, 20, 50) * pi / 180              # angles of the half-planes in the x-z plane
heights <- c(0.55, -0.35, 0.25, -0.6)
radius <- c(1.15, 1.25, 1.1, 1.2)
pts4 <- data.frame(x = radius * cos(theta), y = heights, z = radius * sin(theta), lab = c("c", "d", "e", "f"))
Vp <- rot(60, 40)
half <- function(t, h = 1.45, r = 1.7) rbind(c(0, -h, 0), c(r * cos(t), -h, r * sin(t)), c(r * cos(t), h, r * sin(t)), c(0, h, 0))
quads <- bind_rows(lapply(seq_along(theta), function(i) data.frame(view(half(theta[i]), Vp), plane = i)))
quads$depth_plane <- ave(quads$depth, quads$plane)
# a transversal line meets the four half-planes in four collinear points with the same cross-ratio
a0 <- c(0.85, 1.0, 0); dir <- c(0, -0.12, 1.0)
tcross <- sapply(theta, function(t) { n <- c(-sin(t), 0, cos(t)); -sum(n * a0) / sum(n * dir) })
tp <- t(sapply(tcross, function(s) a0 + s * dir))
tline <- view(rbind(a0 + (min(tcross) - 0.3) * dir, a0 + (max(tcross) + 0.3) * dir), Vp)
axis <- view(rbind(c(0, -1.7, 0), c(0, 1.7, 0)), Vp)
pij <- view(rbind(c(0, -1.0, 0), c(0, 1.0, 0)), Vp)
P4 <- cbind(pts4, view(pts4[, c("x", "y", "z")], Vp)); TP <- view(tp, Vp)
PCOL <- c("#1a73e8", "#188038", "#e8710a", "#d01884")
order_planes <- order(-tapply(quads$depth, quads$plane, mean))
xr <- range(c(quads$u, axis$u, tline$u)) + c(-0.08, 0.08); yr <- range(c(quads$v, axis$v, tline$v)) + c(-0.08, 0.08)
g <- ggplot()
for (i in order_planes) {
  q <- quads[quads$plane == i, ]
  g <- g + geom_polygon(data = q, aes(u, v), fill = PCOL[i], alpha = 0.13, colour = PCOL[i], linewidth = 0.35)
}
g <- g +
  geom_segment(aes(x = axis$u[1], y = axis$v[1], xend = axis$u[2], yend = axis$v[2]), colour = INK, linewidth = 1.2) +
  geom_segment(aes(x = tline$u[1], y = tline$v[1], xend = tline$u[2], yend = tline$v[2]), colour = "#5f6368",
               linewidth = 0.7, linetype = "22") +
  geom_point(data = TP, aes(u, v), shape = 21, fill = "white", colour = PCOL, size = 2.6, stroke = 1.2) +
  geom_point(data = P4, aes(u, v), colour = PCOL, size = 3.4) +
  geom_point(data = pij, aes(u, v), colour = INK, size = 3.6) +
  coord_equal(xlim = xr, ylim = yr, expand = FALSE) + blank + theme(plot.margin = margin(0, 0, 0, 0))
save_png(g, "pencil_3d.png", 4.0, 4.0 * diff(yr) / diff(xr))
lim <- list(x.range = xr, y.range = yr)
# check: the cross-ratio of the four half-planes (from their angles) equals that of the transversal points
plane_cr <- sin(theta[3] - theta[1]) * sin(theta[4] - theta[2]) / (sin(theta[3] - theta[2]) * sin(theta[4] - theta[1]))
if (abs(plane_cr - ((tcross[3] - tcross[1]) * (tcross[4] - tcross[2])) / ((tcross[3] - tcross[2]) * (tcross[4] - tcross[1]))) > 1e-9)
  stop("pencil figure: transversal cross-ratio differs from the plane cross-ratio")
cr <- function(a, b, c, d) ((c - a) * (d - b)) / ((c - b) * (d - a))   # cross-ratio of four collinear parameters
write.csv(data.frame(name = "cross_ratio_transversal", value = cr(tcross[1], tcross[2], tcross[3], tcross[4]),
                     t1 = tcross[1], t2 = tcross[2], t3 = tcross[3], t4 = tcross[4]),
          file.path(OUT, "pencil_transversal.csv"), row.names = FALSE)
write.csv(data.frame(name = c("p_i", "p_j", pts4$lab, paste0("t_", pts4$lab)),
                     fx = (c(pij$u, P4$u, TP$u) - lim$x.range[1]) / diff(lim$x.range),
                     fy = 1 - (c(pij$v, P4$v, TP$v) - lim$y.range[1]) / diff(lim$y.range)),
          file.path(OUT, "pencil_3d_labels.csv"), row.names = FALSE)

# ------------------------------------------------------------------ numbers for the layouts
summ <- read_result("abstract_photo_summary.csv")
val <- function(q) summ$Value[summ$Quantity == q]
fill <- list(
  N_COMMON = format(nrow(frames) / 2, big.mark = ","),
  EXAG = as.character(EXAG),
  PENCIL_CR = sprintf("%.2f", read.csv(file.path(OUT, "pencil_transversal.csv"))$value),
  SWAP_PCT = as.character(round(100 * mean(win$Moved[win$Family == "swap"]))),
  MOVED_PCT = as.character(round(100 * mean(win$Moved[win$Family == "jitter"]))),
  S_CONS = sprintf("%.3f", val("S = Q0.1, consistent")),
  S_INJ = sprintf("%.3f", val("S = Q0.1, injected")),
  N_INJ = format(val("Injected points (8 px jitter on V_B)"), big.mark = ","),
  N_PHOTOS = format(val("Photos registered")),
  N_PENCILS = format(val("Pencil cross-ratios compared"), big.mark = ","))
for (q in summ$Quantity) cat("  summary:", q, "=", summ$Value[summ$Quantity == q], "\n")
for (tpl in list.files("figures/paper", pattern = "\\.template\\.svg$", full.names = TRUE)) {
  s <- paste(readLines(tpl, warn = FALSE), collapse = "\n")
  for (k in names(fill)) s <- gsub(paste0("{{", k, "}}"), fill[[k]], s, fixed = TRUE)
  if (grepl("{{", s, fixed = TRUE)) stop("unfilled placeholder in ", tpl)
  writeLines(s, sub("\\.template\\.svg$", ".svg", tpl))
}
cat("Done.\n")
