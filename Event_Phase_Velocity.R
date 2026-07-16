rm(list = ls())

library(tidyverse)
library(lme4)
library(lmerTest)
library(emmeans)
library(ggpubr)
library(fitdistrplus)
library(ggplot2)

# =========================
# 🎨 THEME
# =========================
raincloud_theme <- theme(
  text = element_text(size = 12),
  axis.title = element_text(size = 12),
  axis.text = element_text(size = 11, color = "black"),
  legend.title = element_text(size = 12),
  legend.text = element_text(size = 11),
  plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
  panel.border = element_blank(),
  panel.grid.major = element_blank(),
  panel.grid.minor = element_blank(),
  axis.line = element_line(color = "black")
)

group_colors <- c("CN" = "#1C3D5A", "GR" = "#A52A2A")

# =========================
# 📂 LOAD DATA
# =========================
df <- read.csv("I:/sailor_moon/DATA/Stat_Rs/FINAL_velocity_event_dataset.csv")

df$Dyad <- as.factor(df$Dyad)
df$Bee_ID <- as.factor(df$Bee_ID)
df$Group <- factor(df$Group, levels = c("CN", "GR"))

# =========================
# 📌 DEFINE RAW VARIABLES
# =========================
phase_cols <- c(
  "Cross_Pre_mm_s", "Cross_Post_mm_s",
  "Contact_Pre_mm_s", "Contact_Post_mm_s"
)

df[phase_cols] <- lapply(df[phase_cols], as.numeric)

# =========================
# 🔁 LONG FORMAT (CLEAN)
# =========================
long <- df %>%
  pivot_longer(
    cols = all_of(phase_cols),
    names_to = "Condition",
    values_to = "Velocity"
  ) %>%
  filter(!is.na(Velocity)) %>%
  mutate(
    Event = ifelse(grepl("Cross", Condition), "Cross", "Contact"),
    Timing = ifelse(grepl("Pre", Condition), "Pre", "Post")
  )

long$Event  <- factor(long$Event, levels = c("Contact", "Cross"))
long$Timing <- factor(long$Timing, levels = c("Pre", "Post"))
long$Condition <- factor(long$Condition, levels = phase_cols)

# =========================
# 📊 PANEL A: EVENT
# =========================
pA <- ggplot(long, aes(Event, Velocity, fill = Group)) +
  geom_violin(trim = FALSE, alpha = 0.5) +
  geom_boxplot(width = 0.15, outlier.shape = NA, alpha = 0.7) +
  geom_jitter(position = position_jitterdodge(0.2), size = 0.8, alpha = 0.4) +
  scale_fill_manual(values = group_colors) +
  theme_classic() +
  raincloud_theme +
  labs(title = "A", x = "", y = "Velocity (mm/s)")

# =========================
# 📊 PANEL B: PRE / POST
# =========================
pB <- ggplot(long, aes(Timing, Velocity, fill = Group)) +
  geom_violin(trim = FALSE, alpha = 0.5) +
  geom_boxplot(width = 0.15, outlier.shape = NA, alpha = 0.7) +
  geom_jitter(position = position_jitterdodge(0.2), size = 0.8, alpha = 0.4) +
  scale_fill_manual(values = group_colors) +
  theme_classic() +
  raincloud_theme +
  labs(title = "B", x = "", y = "Velocity (mm/s)")

# =========================
# 📊 PANEL C: FULL STRUCTURE
# =========================
pC <- ggplot(long, aes(Condition, Velocity, fill = Group)) +
  geom_violin(trim = FALSE, alpha = 0.5) +
  geom_jitter(position = position_jitterdodge(0.2), size = 0.6, alpha = 0.3) +
  scale_fill_manual(values = group_colors) +
  theme_classic() +
  raincloud_theme +
  theme(axis.text.x = element_text(angle = 30, hjust = 1)) +
  labs(title = "C", x = "", y = "Velocity (mm/s)")

# =========================
# 📊 PANEL D: DYAD MEANS (SAFE)
# =========================
dyad <- long %>%
  group_by(Dyad, Group, Condition) %>%
  summarise(Velocity = mean(Velocity, na.rm = TRUE), .groups = "drop")

dyad$Condition <- factor(dyad$Condition, levels = phase_cols)

pD <- ggplot(dyad, aes(Condition, Velocity, group = Dyad)) +
  geom_line(aes(color = Group), alpha = 0.4) +
  geom_point(aes(color = Group), size = 2) +
  scale_color_manual(values = group_colors) +
  theme_classic() +
  raincloud_theme +
  theme(axis.text.x = element_text(angle = 30, hjust = 1)) +
  labs(title = "D", x = "", y = "Velocity (mm/s)")

# =========================
# 🧩 FINAL FIGURE
# =========================
final_fig <- ggarrange(pA, pB, pC, pD, ncol = 2, nrow = 2)
final_fig

# =========================
# 📊 DISTRIBUTION CHECK
# =========================
plotdist(long$Velocity, histo = TRUE, demp = TRUE)

fit.norm  <- fitdist(long$Velocity, "norm")
fit.gamma <- fitdist(long$Velocity, "gamma")

denscomp(list(fit.norm, fit.gamma))
gofstat(list(fit.norm, fit.gamma))

# =========================
# 🧠 GLMM (GAMMA LOG)
# =========================
model <- glmer(Velocity ~ Group * Event + (1 | Bee_ID),
               family = Gamma(link = "log"),
               data = long)

summary(model)

# NOTE: singular fit is OK (random effect ~0 variance)

# =========================
# 📌 POSTHOC
# =========================
emmeans(model, pairwise ~ Group)
emmeans(model, pairwise ~ Event)
emmeans(model, pairwise ~ Group | Event)

# =========================
# 📈 MODEL-BASED PLOT
# =========================
emm <- emmeans(model, ~ Group * Event)
emm_df <- as.data.frame(emm)

ggplot(emm_df, aes(Event, emmean, color = Group, group = Group)) +
  geom_point(size = 3) +
  geom_line(linewidth = 1) +
  geom_errorbar(aes(ymin = asymp.LCL, ymax = asymp.UCL),
                width = 0.1) +
  scale_color_manual(values = group_colors) +
  theme_classic() +
  raincloud_theme +
  labs(title = "Model estimated means (GLMM)",
       y = "Velocity (log scale)", x = "")
