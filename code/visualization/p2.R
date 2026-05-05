# Figure 2: Mechanisms underlying mindset effects
# Data: belief_grid simulation

library(ggplot2)
library(dplyr)
library(tidyr)
library(bruceR)
library(patchwork)

# ----- config -----
simulation_id <- "belief_grid_20260505_011515"
data_file     <- sprintf("results/simulation/%s/trial_results.csv", simulation_id)

plot_dir  <- "results/plots"
dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)

colormap_skill <- c(
  "Harvest_Easy"          = "#BFD7EA",
  "Cultivate_Easy"        = "#4C78A8",
  "Harvest_Challenging"   = "#F2C6A0",
  "Cultivate_Challenging" = "#D17C45"
)

colormap_contrast <- c("Easy" = "#4C78A8", "Challenging" = "#D17C45")

prior_threshold <- c(0.3, 0.7)

# ----- load & parse -----
df <- read.csv(data_file) %>%
  separate(agent, into = c("m0m", "m0c"), sep = "_", convert = TRUE)

# ----- Panel A & B: expected return and switch point at t=0 -----
df_t0 <- df %>%
  group_by(env, M_level, m0m, m0c, sim) %>%
  slice_min(trial, n = 1) %>%
  ungroup()

df_arm_summary <- df_t0 %>%
  select(env, M_level, m0m, m0c, sim,
         skill1_return, skill2_return,
         skill1_switch_point, skill2_switch_point) %>%
  pivot_longer(
    cols = c(skill1_return, skill2_return, skill1_switch_point, skill2_switch_point),
    names_to = c("skill", "metric"),
    names_pattern = "(skill[12])_(.*)"
  ) %>%
  mutate(skill = recode(skill, skill1 = "Easy", skill2 = "Challenging")) %>%
  group_by(M_level, m0m, skill, metric) %>%
  summarise(mean_value = mean(value, na.rm = TRUE), .groups = "drop") %>%
  filter(M_level == 0.5)

plot_arm_metric <- function(data, metric_name, title) {
  data %>%
    filter(metric == metric_name) %>%
    ggplot(aes(x = m0m, y = mean_value, color = skill, group = skill)) +
    geom_vline(xintercept = c(0.3, 0.5, 0.7), linetype = "dashed", color = "grey70", linewidth = 0.4) +
    geom_line() +
    geom_point(size = 1) +
    scale_x_continuous(
      breaks = c(0, 0.3, 0.5, 0.7, 1),
      labels = c("0", "0.3", "0.5", "0.7", "1")
    ) +
    scale_color_manual(values = colormap_contrast) +
    labs(
      x     = bquote("Malleability Prior" ~ hat(b)[0] == alpha / (alpha + beta)),
      y     = NULL,
      title = title,
      color = "Skill"
    ) +
    theme_bruce() +
    theme(
      strip.background = element_blank(),
      plot.title       = element_text(size = 11),
      axis.title.x     = element_text(size = 11),
      axis.title.y     = element_blank(),
      legend.title     = element_text(size = 10)
    )
}

p2a <- plot_arm_metric(df_arm_summary, "return",
                       bquote("Expected Return Calculated at" ~ t == 0))
p2b <- plot_arm_metric(df_arm_summary, "switch_point",
                       bquote("Optimal Switch Point Calculated at" ~ t == 0))

# ----- Panel C: action dynamics over trials -----
# fixed prior strength = 30, three prior means: 0.3, 0.5, 0.7
m0m_list <- c(0.3, 0.5, 0.7)

p2c <- df %>%
  filter(round(m0c) == 30, round(m0m, 1) %in% m0m_list) %>%
  mutate(
    move = paste(str_to_title(action), intended_skill, sep = "_")
  ) %>%
  group_by(env, M_level, m0m, trial, move) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(env, M_level, m0m, trial) %>%
  mutate(proportion = n / sum(n)) %>%
  ungroup() %>%
  mutate(
    M_level_label = factor(M_level, levels = c(0.7, 0.5, 0.3),
                           labels = c("High Malleability\n(m=0.7)",
                                      "Moderate Malleability\n(m=0.5)",
                                      "Low Malleability\n(m=0.3)")),
    # double-line row label: "Low/Mid/High Prior" + hat(b)[0]==value
    m0m_group = case_when(
      m0m >= prior_threshold[2] ~ "High Prior",
      m0m <= prior_threshold[1] ~ "Low Prior",
      TRUE                      ~ "Mid Prior"
    ),
    m0m_label = sprintf(
      'atop("%s", hat(b)[0]==%.1f)',
      m0m_group,
      round(m0m, 2)
    ),
    m0m_label = factor(
      m0m_label,
      levels = sprintf(
        'atop("%s", hat(b)[0]==%.1f)',
        case_when(
          round(m0m_list, 2) >= prior_threshold[2] ~ "High Prior",
          round(m0m_list, 2) <= prior_threshold[1] ~ "Low Prior",
          TRUE                                      ~ "Mid Prior"
        ),
        round(m0m_list, 2)
      )
    )
  ) %>%
  ggplot(aes(x = trial, y = proportion, fill = move, color = move)) +
  geom_bar(stat = "identity", position = "stack") +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  scale_fill_manual(values  = colormap_skill, labels = function(x) gsub("_", " ", x)) +
  scale_color_manual(values = colormap_skill, guide  = "none") +
  facet_grid(rows = vars(m0m_label), cols = vars(M_level_label),
             labeller = labeller(m0m_label = label_parsed)) +
  guides(fill = guide_legend(nrow = 2, byrow = TRUE, title = "Action")) +
  labs(x = "Time Step", y = "Action Proportion per Time Step") +
  theme_bruce() +
  theme(
    strip.background  = element_blank(),
    axis.ticks.y      = element_blank(),
    axis.text.y       = element_blank(),
    legend.position   = "bottom",
    legend.title      = element_text(size = 10),
    axis.title.x      = element_text(size = 12),
    axis.title.y      = element_text(size = 12),
    legend.box.margin = margin(t = -10, b = 0),
    legend.margin     = margin(t = -5, b = 0)
  )

# ----- assemble -----
p2_top <- (p2a + p2b) +
  plot_layout(guides = "collect") &
  theme(
    legend.position   = "bottom",
    legend.box.margin = margin(t = -10, b = 0),
    legend.margin     = margin(t = -5, b = 0)
  )

p2 <- p2_top / p2c +
  plot_annotation(tag_levels = "A") +
  plot_layout(heights = c(1, 2.5))

ggsave(file.path(plot_dir, "p2.pdf"), p2, width = 1350, height = 1400, dpi = 200, units = "px")
