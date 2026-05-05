# Figure 3: Boundary conditions of mindset effects
# Data: environment_grid simulation

library(ggplot2)
library(dplyr)
library(tidyr)
library(bruceR)

# ----- config -----
simulation_id <- "environment_grid_20260505_012349"

data_file <- sprintf("results/simulation/%s/trial_results.csv", simulation_id)
plot_dir  <- "results/plots"
dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)

plt_colors <- c("Fixed Mindset" = "#E7A539", "Growth Mindset" = "#74BAD3")

metric_labels <- c(
  challenge_seeking = "Challenge-Seeking",
  persistence       = "Persistence"
)

param_labels <- c(
  R = "Reward Baseline R\n(Ascending)",
  W = "Reward Growth Rate w\n(Descending)",
  T = "Time Horizon T\n(Descending)"
)

# ----- load and compute metrics -----
df_all <- read.csv(data_file)

df_metrics <- df_all %>%
  group_by(env, M_level, agent, sim) %>%
  summarise(
    persistence       = mean(action == "cultivate"),
    challenge_seeking = mean(intended_skill == "Challenging"),
    .groups = "drop"
  ) %>%
  separate(env, into = c("param", "env_value"), sep = "=") %>%
  mutate(
    env_value = as.numeric(env_value),
    x_plot    = if_else(param %in% c("W", "T"), -env_value, env_value)
  )

df_summary <- df_metrics %>%
  pivot_longer(cols = c(challenge_seeking, persistence),
               names_to = "metric", values_to = "value") %>%
  group_by(param, env_value, x_plot, metric, agent) %>%
  summarise(
    mean_value = mean(value, na.rm = TRUE),
    sd         = sd(value, na.rm = TRUE),
    .groups    = "drop"
  ) %>%
  ungroup() %>%
  mutate(
    metric = factor(metric, levels = c("challenge_seeking", "persistence")),
    agent  = recode(agent, gm = "Growth Mindset", fm = "Fixed Mindset"),
    agent  = factor(agent, levels = c("Fixed Mindset", "Growth Mindset"))
  )

# grey rectangle marking the moderate reference environment used in other analyses
df_rect <- df_summary %>%
  select(param, x_plot) %>%
  distinct() %>%
  group_by(param) %>%
  summarise(
    half_step = abs(max(x_plot) - min(x_plot)) / (n() - 1) / 2,
    xmin      = median(x_plot) - half_step,
    xmax      = median(x_plot) + half_step,
    .groups   = "drop"
  )

# ----- plot -----


p3 <- ggplot(df_summary,
             aes(x = x_plot, y = mean_value, color = agent, group = agent)) +
  geom_rect(data = df_rect,
            aes(xmin = xmin, xmax = xmax, ymin = -Inf, ymax = Inf),
            inherit.aes = FALSE, fill = "#e0e0e0", alpha = 0.5) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 1.2) +
  geom_errorbar(aes(ymin = mean_value - sd, ymax = mean_value + sd), width = 0) +
  scale_color_manual(
    values = plt_colors,
    labels = c(
      bquote("Fixed Mindset (" * hat(b)[0] == 0.4 * ")"),
      bquote("Growth Mindset (" * hat(b)[0] == 0.6 * ")")
    )
  ) +
  scale_x_continuous(labels = function(x) abs(x), n.breaks = 5, expand = c(0.1, 0.1)) +
  scale_y_continuous(expand = c(0.03, 0.03)) +
  facet_grid(rows = vars(metric), cols = vars(param),
             scales = "free",
             labeller = labeller(metric = as_labeller(metric_labels),
                                 param  = as_labeller(param_labels))) +
  labs(x = "Environment Parameter", y = NULL, color = "Agent") +
  theme_bruce() +
  theme(
    strip.background  = element_blank(),
    legend.position   = "bottom",
    legend.box.margin = margin(t = -10),
    legend.margin     = margin(t = -5)
  )

ggsave(file.path(plot_dir, "p3.pdf"), p3, width = 1400, height = 900, dpi = 200, units = "px")