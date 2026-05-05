# Figure 1: Behavioral outcomes across mindset priors
# Data: belief_grid simulation

library(ggplot2)
library(dplyr)
library(tidyr)
library(bruceR)

# ----- config -----
simulation_id <- "belief_grid_20260505_011515"
data_file     <- sprintf("results/simulation/%s/trial_results.csv", simulation_id)

plot_dir  <- "results/plots"
dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)

# ----- load & parse -----
df <- read.csv(data_file) %>%
  separate(agent, into = c("m0m", "m0c"), sep = "_", convert = TRUE)

# ----- metrics per simulation run -----
df_belief <- df %>%
  group_by(env, M_level, m0m, m0c, sim) %>%
  summarise(
    belief_updating = (
      (last(skill1_Mhat) - first(skill1_Mhat)) +
        (last(skill2_Mhat) - first(skill2_Mhat))
    ) / 2,
    .groups = "drop"
  )

df_metrics <- df %>%
  group_by(env, M_level, m0m, m0c, sim) %>%
  summarise(
    reward            = sum(reward),
    persistence       = mean(action == "cultivate"),
    challenge_seeking = mean(intended_skill == "Challenging"),
    .groups = "drop"
  ) %>%
  left_join(df_belief, by = c("env", "M_level", "m0m", "m0c", "sim"))

# ----- summary across sims -----
df_summary <- df_metrics %>%
  pivot_longer(cols = c(challenge_seeking, persistence, reward, belief_updating),
               names_to = "metric", values_to = "value") %>%
  group_by(M_level, m0m, m0c, metric) %>%
  summarise(
    mean_value = mean(value, na.rm = TRUE),
    se         = sd(value, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  ) %>%
  mutate(
    metric = factor(metric, levels = c("challenge_seeking", "persistence", "reward", "belief_updating")),
    M_level_label = factor(M_level, levels = c(0.7, 0.5, 0.3),
                           labels = c("High Malleability\n(m=0.7)",
                                      "Moderate Malleability\n(m=0.5)",
                                      "Low Malleability\n(m=0.3)"))
  )

# ----- facet row labels with formulas (label_parsed compatible) -----
metric_labels <- c(
  challenge_seeking = 'atop("Challenge-Seeking", N[Challenging]/T)',
  persistence       = 'atop("Persistence", N[Cultivate]/T)',
  reward            = 'atop("Cumulative Reward", sum(r[t]))',
  belief_updating   = 'atop("Belief Updating", hat(b)[T]-hat(b)[0])'
)

# ----- per-column vline data -----
df_vline <- data.frame(
  M_level_label = factor(
    c("High Malleability\n(m=0.7)", "Moderate Malleability\n(m=0.5)", "Low Malleability\n(m=0.3)"),
    levels = c("High Malleability\n(m=0.7)", "Moderate Malleability\n(m=0.5)", "Low Malleability\n(m=0.3)")
  ),
  xintercept = c(0.7, 0.5, 0.3)
)

# ----- plot -----
p1 <- ggplot(df_summary,
             aes(x = m0m, y = mean_value, color = factor(m0c), group = factor(m0c))) +
  geom_vline(data = df_vline,
             aes(xintercept = xintercept),
             linetype = "dashed", color = "grey30", alpha = 0.6) +
  geom_line() +
  geom_point() +
  facet_grid(rows = vars(metric), cols = vars(M_level_label),
             scales = "free_y",
             labeller = labeller(metric = as_labeller(metric_labels, label_parsed))) +
  scale_color_brewer(palette = "Blues", direction = 1) +
  scale_x_continuous(
    labels = function(x) sub("\\.?0+$", "", formatC(x, format = "f", digits = 2))
  ) +
  scale_y_continuous(sec.axis = dup_axis(labels = NULL)) +
  labs(
    x     = bquote("Prior Malleability Estimate" ~ hat(b)[0] == alpha / (alpha + beta)),
    y     = NULL,
    color = bquote("Prior Strength " * "(" * alpha + beta * ")")
  ) +
  guides(color = guide_legend(nrow = 1, byrow = TRUE)) +
  theme_bruce() +
  theme(
    strip.background      = element_blank(),
    axis.title.x          = element_text(size = 12),
    axis.title.y.left     = element_blank(),
    axis.ticks.y          = element_blank(),
    legend.position       = "bottom",
    legend.title          = element_text(size = 10),
    legend.key.height     = unit(0.5, "lines"),
    legend.box.margin     = margin(t = -6, b = 0),
    legend.margin         = margin(t = 0,  b = 0)
  )

ggsave(file.path(plot_dir, "p1.pdf"), p1, width = 1250, height = 1350, dpi = 200, units = "px")
