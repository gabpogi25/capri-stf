# figures/abstracts/render_assets.R
# Renders the image assets of the graphical abstracts from the photo demo results
# (experiments/run_abstract_photo_demo.py -> experiments/results/abstract_photo_*.csv).
# Output: figures/abstracts/assets/*.png (transparent where useful). The layouts that place
# these assets are the Inkscape SVGs in figures/abstracts/ (build.sh exports them).
# Run from the repository root: Rscript figures/abstracts/render_assets.R

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(jpeg); library(grid); library(magick)
})

RES <- "experiments/results"
OUT <- "figures/abstracts/assets"
PHOTOS <- "figures/abstracts/photos/sacre_coeur"
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
read_result <- function(f) read.csv(file.path(RES, f), comment.char = "#")

COL_A <- "#1f77ff"; COL_B <- "#ff2a1a"; COL_OK <- "#2bd92b"; COL_BAD <- "#ff2a1a"
FONT <- "Roboto"

model <- read_result("abstract_photo_model.csv")
cams <- read_result("abstract_photo_cameras.csv")
obs <- read_result("abstract_photo_observations.csv")
dist <- read_result("abstract_photo_discrepancies.csv")
summ <- read_result("abstract_photo_summary.csv")

# ------------------------------------------------------------------ virtual view
pts <- model %>% filter(Track >= 3)
centre <- apply(pts[, c("X", "Y", "Z")], 2, median)
radius <- sqrt(rowSums(sweep(as.matrix(pts[, c("X", "Y", "Z")]), 2, centre)^2))
pts <- pts[radius < quantile(radius, 0.92), ]
view_matrix <- function(azimuth, elevation) {
  a <- azimuth * pi / 180; e <- elevation * pi / 180
  # COLMAP world: y points down. Build a camera looking at the centre.
  Ry <- matrix(c(cos(a), 0, sin(a), 0, 1, 0, -sin(a), 0, cos(a)), 3, byrow = TRUE)
  Rx <- matrix(c(1, 0, 0, 0, cos(e), -sin(e), 0, sin(e), cos(e)), 3, byrow = TRUE)
  Rx %*% Ry
}
project_view <- function(xyz, V) {                 # orthographic view of the model
  p <- t(V %*% (t(as.matrix(xyz)) - centre))
  data.frame(u = p[, 1], v = -p[, 2], depth = p[, 3])
}

# Four of the photos are telephoto shots taken about 10 model units away, against a model
# radius below 2. For display only, camera centres farther than `cap` from the model centre
# are moved along the line to the centre until they are `cap` away (orientation unchanged).
frustum_lines <- function(cams, V, scale = 0.45, cap = 3.2) {
  out <- list()
  for (k in seq_len(nrow(cams))) {
    cm <- cams[k, ]
    R <- matrix(unlist(cm[paste0("R", rep(0:2, each = 3), rep(0:2, 3))]), 3, byrow = TRUE)
    C <- c(cm$Cx, cm$Cy, cm$Cz)
    d <- sqrt(sum((C - centre)^2))
    if (d > cap) C <- centre + (C - centre) * cap / d
    corners <- rbind(c(0, 0), c(cm$Width, 0), c(cm$Width, cm$Height), c(0, cm$Height))
    rays <- cbind((corners[, 1] - cm$Px) / cm$Focal, (corners[, 2] - cm$Py) / cm$Focal, 1)
    world <- t(C + t(rays %*% R) * scale)            # R' * ray, scaled
    P <- project_view(rbind(C, world), V)
    seg <- data.frame(x = P$u[c(1, 1, 1, 1, 2, 3, 4, 5)], y = P$v[c(1, 1, 1, 1, 2, 3, 4, 5)],
                      xend = P$u[c(2, 3, 4, 5, 3, 4, 5, 2)], yend = P$v[c(2, 3, 4, 5, 3, 4, 5, 2)])
    seg$Subset <- cm$Subset; seg$Image <- cm$Image
    out[[k]] <- seg
  }
  bind_rows(out)
}

render_cloud <- function(V, file, width = 2400, height = 1800, frustum_scale = 0.38, point_size = 1.5) {
  P <- cbind(pts, project_view(pts[, c("X", "Y", "Z")], V))
  P <- P[order(-P$depth), ]
  P$col <- rgb(P$R, P$G, P$B, maxColorValue = 255)
  fr <- frustum_lines(cams, V, frustum_scale)
  g <- ggplot() +
    geom_point(data = P, aes(u, v), colour = P$col, size = point_size, alpha = 0.95) +
    geom_segment(data = fr, aes(x, y, xend = xend, yend = yend, colour = Subset), linewidth = 1.3, lineend = "round") +
    scale_colour_manual(values = c(A = COL_A, B = COL_B), guide = "none") +
    coord_equal() + theme_void() +
    theme(plot.background = element_rect(fill = "transparent", colour = NA),
          panel.background = element_rect(fill = "transparent", colour = NA))
  ggsave(file.path(OUT, file), g, width = width / 300, height = height / 300, dpi = 300, bg = "transparent")
}

V <- view_matrix(azimuth = 0, elevation = 25)           # from behind and above the cameras, facing the facade
render_cloud(V, "cloud_cameras.png")

# ------------------------------------------------------------------ photo overlays
# Saved at the native pixel size of each photo. The plain photos are linked from photos/ directly.
photo_overlay <- function(image, file, mode = c("injected", "score"), point_size = 1.7) {
  mode <- match.arg(mode)
  img <- readJPEG(file.path(PHOTOS, image))
  h <- dim(img)[1]; w <- dim(img)[2]
  o <- obs %>% filter(Image == image)
  g <- ggplot() + annotation_raster(img, 0, w, -h, 0) +
    coord_equal(xlim = c(0, w), ylim = c(-h, 0), expand = FALSE) + theme_void()
  if (mode == "injected") {                         # ground truth of the illustration
    o$Injected <- as.logical(o$Injected)
    g <- g + geom_point(data = o, aes(x, -y, colour = Injected), size = point_size, alpha = 0.95) +
      scale_colour_manual(values = c(`FALSE` = COL_OK, `TRUE` = COL_BAD), guide = "none")
  } else {                                          # output of CAPRI: per-point pencil score, as a percentile
    o$Percentile <- rank(o$Pencil_Score) / nrow(o)
    o <- o[order(o$Percentile), ]
    g <- g + geom_point(data = o, aes(x, -y, colour = Percentile), size = point_size, alpha = 0.95) +
      scale_colour_gradientn(colours = c(COL_OK, "#ffd21a", COL_BAD), guide = "none")
  }
  ggsave(file.path(OUT, file), g, width = w / 300, height = h / 300, dpi = 300)
}

showcase_a <- "71295362_4051449754.jpg"            # the V_A / V_B pair with the most shared points
showcase_b <- "93341989_396310999.jpg"
photo_overlay(showcase_a, "photo_A_points.png")
photo_overlay(showcase_b, "photo_B_points.png")
photo_overlay(showcase_a, "photo_A_scores.png", "score")

# ------------------------------------------------------------------ match pair (hloc / SuperGlue style)
match_pair <- function(ia, ib, file, gap = 24, every = 3) {
  A <- readJPEG(file.path(PHOTOS, ia)); B <- readJPEG(file.path(PHOTOS, ib))
  ha <- dim(A)[1]; wa <- dim(A)[2]; hb <- dim(B)[1]; wb <- dim(B)[2]
  sb <- ha / hb                                        # scale B to A's height
  oa <- obs %>% filter(Image == ia) %>% select(Point, xa = x, ya = y, Injected)
  ob <- obs %>% filter(Image == ib) %>% select(Point, xb = x, yb = y)
  m <- inner_join(oa, ob, by = "Point") %>% mutate(xb = wa + gap + xb * sb, yb = yb * sb)
  m$Injected <- as.logical(m$Injected)
  m <- m[order(m$Point), ][seq(1, nrow(m), by = every), ]   # every third shared point, for legibility
  g <- ggplot() +
    annotation_raster(A, 0, wa, -ha, 0) +
    annotation_raster(B, wa + gap, wa + gap + wb * sb, -ha, 0) +
    geom_segment(data = m, aes(xa, -ya, xend = xb, yend = -yb, colour = Injected), linewidth = 0.45, alpha = 0.9) +
    geom_point(data = m, aes(xa, -ya, colour = Injected), size = 1.3) +
    geom_point(data = m, aes(xb, -yb, colour = Injected), size = 1.3) +
    scale_colour_manual(values = c(`FALSE` = COL_OK, `TRUE` = COL_BAD), guide = "none") +
    coord_equal(xlim = c(0, wa + gap + wb * sb), ylim = c(-ha, 0), expand = FALSE) + theme_void()
  ggsave(file.path(OUT, file), g, width = (wa + gap + wb * sb) / 300, height = ha / 300, dpi = 300, bg = "white")
  nrow(m)
}
n_pair <- match_pair(showcase_a, showcase_b, "match_pair.png")

# ------------------------------------------------------------------ pencil of planes on a photo
pencil_on_photo <- function(image, file) {
  img <- readJPEG(file.path(PHOTOS, image)); h <- dim(img)[1]; w <- dim(img)[2]
  o <- obs %>% filter(Image == image)
  # line p_i p_j: two points far apart vertically near the middle of the facade
  mid <- o %>% filter(abs(x - median(x)) < 0.12 * w)
  pi_ <- mid[which.min(mid$y), ]; pj <- mid[which.max(mid$y), ]
  others <- o %>% filter(!(Point %in% c(pi_$Point, pj$Point)))
  # four points well away from the line: two on each side, at different heights
  dx <- pj$x - pi_$x; dy <- pj$y - pi_$y; L <- sqrt(dx^2 + dy^2)
  side <- ((others$x - pi_$x) * dy - (others$y - pi_$y) * dx) / L      # signed distance to the line
  along <- ((others$x - pi_$x) * dx + (others$y - pi_$y) * dy) / L^2     # position along the line
  pick <- c(which.max(ifelse(side < -0.18 * w & along > 0.55, -side, -Inf)),
            which.max(ifelse(side < -0.10 * w & along < 0.35, -side, -Inf)),
            which.max(ifelse(side > 0.10 * w & along < 0.35, side, -Inf)),
            which.max(ifelse(side > 0.18 * w & along > 0.55, side, -Inf)))
  quad <- others[pick, ]
  cols <- c("#8a4f7d", "#3a6ea5", "#2a7f62", "#b5541a")
  tri <- bind_rows(lapply(1:4, function(k) data.frame(
    x = c(pi_$x, pj$x, quad$x[k]), y = c(pi_$y, pj$y, quad$y[k]), k = k)))
  g <- ggplot() + annotation_raster(img, 0, w, -h, 0) +
    geom_polygon(data = tri, aes(x, -y, group = k, fill = factor(k)), alpha = 0.33, colour = NA) +
    geom_polygon(data = tri, aes(x, -y, group = k, colour = factor(k)), fill = NA, linewidth = 0.8) +
    geom_segment(aes(x = pi_$x, y = -pi_$y, xend = pj$x, yend = -pj$y), colour = "white", linewidth = 2.2) +
    geom_point(data = quad, aes(x, -y), colour = "white", fill = cols, shape = 21, size = 4.5, stroke = 1.2) +
    geom_point(data = rbind(pi_, pj), aes(x, -y), colour = "white", fill = "black", shape = 21, size = 5, stroke = 1.4) +
    scale_fill_manual(values = cols, guide = "none") + scale_colour_manual(values = cols, guide = "none") +
    coord_equal(xlim = c(0, w), ylim = c(-h, 0), expand = FALSE) + theme_void()
  ggsave(file.path(OUT, file), g, width = w / 300, height = h / 300, dpi = 300)
}
pencil_on_photo(showcase_a, "pencil_on_photo.png")

# ------------------------------------------------------------------ discrepancy distributions
q <- dist %>% group_by(Version) %>% summarise(Q = quantile(D, 0.1))
g <- ggplot(dist, aes(log10(D), fill = Version, colour = Version)) +
  geom_density(aes(alpha = Version), linewidth = 1.1, adjust = 1.2) +
  scale_alpha_manual(values = c(consistent = 0.45, injected = 0.12), guide = "none") +
  coord_cartesian(xlim = c(-3.5, 1)) +
  geom_vline(data = q, aes(xintercept = log10(Q), colour = Version), linewidth = 1.2, linetype = "22") +
  scale_fill_manual(values = c(consistent = COL_OK, injected = COL_BAD), guide = "none") +
  scale_colour_manual(values = c(consistent = "#1a9e1a", injected = COL_BAD), guide = "none") +
  labs(x = expression(log[10]~D[t]), y = NULL) +
  theme_classic(base_size = 22, base_family = FONT) +
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank(), axis.line.y = element_blank(),
        plot.background = element_rect(fill = "transparent", colour = NA),
        panel.background = element_rect(fill = "transparent", colour = NA))
ggsave(file.path(OUT, "discrepancy_density.png"), g, width = 6, height = 3.2, dpi = 300, bg = "transparent")

# ------------------------------------------------------------------ fill the layout templates
# The layouts are hand-made Inkscape SVGs (*.template.svg). Every number in them is a double-brace
# placeholder filled here from the result files; build.sh then exports the filled SVGs.
value <- function(q) summ$Value[summ$Quantity == q]
rt <- read_result("pencil_runtime.csv")
ms <- function(stage, n, impl) rt$Median_ms[rt$Stage == stage & rt$N_Points == n & rt$Implementation == impl]
fmt_ms <- function(x) if (x >= 1000) sprintf("%.1f s", x / 1000) else if (x >= 100) sprintf("%.0f ms", x) else sprintf("%.1f ms", x)
fmt_int <- function(x) format(round(x), big.mark = ",", scientific = FALSE)
values <- c(
  N_PHOTOS = fmt_int(value("Photos registered")),
  N_MODEL = fmt_int(value("Model points")),
  REPROJ = sprintf("%.2f", value("Mean reprojection error (px)")),
  N_COMMON = fmt_int(value("Points seen twice in each subset")),
  N_INJ = fmt_int(value("Injected points (8 px jitter on V_B)")),
  N_PENCILS = fmt_int(value("Pencil cross-ratios compared")),
  S_CONS = sprintf("%.3f", value("S = Q0.1, consistent")),
  S_INJ = sprintf("%.3f", value("S = Q0.1, injected")),
  SCORE_INJ = sprintf("%.2f", value("Median point score, injected points")),
  SCORE_OK = sprintf("%.2f", value("Median point score, other points")),
  N_PAIR = fmt_int(n_pair),
  KAPPA_CPP = fmt_ms(ms("kappa_batch", 50, "cpp_batch")),
  KAPPA_NP = fmt_ms(ms("kappa_batch", 50, "numpy_batch")),
  KAPPA_LOOP = fmt_ms(ms("kappa_batch", 50, "python_loop")),
  SEL_BATCH = fmt_ms(ms("selector", 30, "batch_scoring")),
  SEL_TUPLE = fmt_ms(ms("selector", 30, "per_tuple_scoring")),
  PEN12 = fmt_ms(ms("pencil_score", 12, "cpp")),
  PEN300 = fmt_ms(ms("pencil_score", 300, "cpp")))
for (tpl in list.files("figures/abstracts", pattern = "[.]template[.]svg$", full.names = TRUE)) {
  svg <- paste(readLines(tpl, warn = FALSE), collapse = "\n")
  for (k in names(values)) svg <- gsub(paste0("{{", k, "}}"), values[[k]], svg, fixed = TRUE)
  left <- regmatches(svg, gregexpr("[{][{][A-Z0-9_]+[}][}]", svg))[[1]]
  if (length(left)) stop("unfilled placeholders in ", tpl, ": ", paste(unique(left), collapse = ", "))
  writeLines(svg, sub("[.]template[.]svg$", ".svg", tpl))
}
print(values)
